/**
 * @file    main.js
 *          Used for training over the DCP network.
 *          Adapted from the tensorflow.js mnist-core example.
 * @author  Ian Chew
 * @date    Jan 2019
 */

import {MnistData} from './data.js';
import {MnistModel} from './mnist.js';
import {worker} from './worker.js';
import {TRAIN_STEPS, BATCH_SIZE, WORKER_COUNT} from './config.js';
/* globals Generator, alert */
//import Compute from '/compute.js' //eslint-disable-line

/**
 * Load in the images and corresponding labels.
 */
async function load() {
	const data = new MnistData();
	await data.load();
	return data;
}

/**
 * Returns the worker string, containing the code the worker will execute.
 *
 * @returns {string} - The code the worker will execute.
 */
function getWorkerString() {
	const workerstr = '' + worker;
	// Insert after const tf = require('tf.min.js');
	const entryPoint = workerstr.indexOf(';') + 1;
	return workerstr.slice(0, entryPoint) + MnistModel + workerstr.slice(entryPoint);
}

/**
 * Train the model by sending out jobs over the network.
 *
 * @param {MnistData} - Object containing the data to train on.
 * @returns {MnistModel} - The model that was trained.
 */
async function train(data) {
	// Initialize our neural network model.
	const model = new MnistModel();
	// Get the code to send to the workers.
	const workerstr = getWorkerString();

	for (let i=0; i<TRAIN_STEPS; ++i) {
		document.getElementById('batch').innerHTML = 'Batch ' + (i + 1);
		// Array of data to send to the workers.
		const batchlist = []
		for (let j=0; j<WORKER_COUNT; ++j) {
			const batch = data.nextTrainBatch(BATCH_SIZE);
			batchlist.push({
				images: batch.xs,
				labels: batch.labels
			});
		}
		const model_weights = model.toArray();

		// Number of workers complete so far.
		let workers_done = 0;

		// compute.for takes in an array of inputs, one element of it will
		// be send to each worker, the code for the worker to execute, and
		// any additional arguments to call the worker function with, which
		// will be the same for all the workers.
		const gen = compute.for([batchlist], workerstr, [model_weights]);
		gen.requires('tensorflowdcp/tf.min.js');
		gen.on('result', () => {
			++workers_done;
			document.getElementById('workers').innerHTML =
			    'Batch progress: ' + workers_done + '/' + WORKER_COUNT;
		});
		gen.on('complete', () => console.log('Batch ' + (i + 1) + ' complete.'));
		// Name that appears in the worker's browser.
		gen._generator.public = {
			name: 'MNIST Batch ' + (i + 1)
		};

		// await the results:
		// const res = await gen.exec(0.0001 * WORKER_COUNT);
		const res = await gen.localExec();
		// Turn the gradients back into tensors.
		const tensorGrads = decodeGradients(res, model);
		// Average the gradients between the workers.
		const avgGrads = averageGradients(tensorGrads);
		// Update the model's weights using the gradients.
		model.applyGradients(avgGrads);

		// Free up the allocated gradient tensors.
		tf.dispose(tensorGrads);
		tf.dispose(avgGrads);

		// Test and print out accuracy...
		test(model, data);
	}
	return model;
}

/**
 * Test the model, returning the accuracy on the test dataset.
 *
 * @param {MnistModel} model - The model to test.
 * @param {MnistData} data - The dataset to test on.
 * @returns {number} - The accuracy of the model from 0 to 1.
 */
async function test(model, data) {
	const testExamples = 1024;
	const batch = data.nextTestBatch(testExamples);
	// Convert batch to tensors:
	batch.xs = tf.tensor2d(batch.xs, [testExamples, model.IMAGE_SIZE*model.IMAGE_SIZE]);
	batch.labels = tf.tensor2d(batch.labels, [testExamples, model.LABELS_SIZE]);
	const predictions = model.predict(batch.xs);
	const labels = model.classesFromLabel(batch.labels);
	const accuracy = getAccuracy(predictions, labels);
	console.log('accuracy = ' + accuracy);
	document.getElementById('acc').innerHTML = 'Accuracy = ' + accuracy;
//	ui.showTestResults(batch, predictions, labels);
}


/**
 * Compares predictions and labels to get the accuracy of the model.
 *
 * @param {Array} predictions - The predicted values, calculated by the model.
 * @param {Array} labels - The true results, from the dataset.
 * @returns {number} - The accuracy from 0 to 1.
 */
function getAccuracy(predictions, labels) {
	let numCorrect = 0;
	for (let i=0; i<predictions.length; ++i) {
		if (predictions[i] == labels[i]) {
			++numCorrect;
		}
	}
	return numCorrect/predictions.length;
}


/**
 * Decode the gradients received from the workers, converting them
 * to tensors.
 *
 * @param {Array} grads - An array of gradients, which themselves are maps
 * from number identifying the part of the model they apply to, to arrays.
 */
function decodeGradients(grads, model) {
	const ret = [];
	for (let grad of grads) {
		ret.push(model.decode(grad));
	}
	return ret;
}

/**
 * Average out the gradients received from the workers.
 *
 * @param {Array} grads - An array of gradients, which themselves are maps
 * from number identifying the part of the model they apply to, to arrays.
 */
function averageGradients(grads) {
	const ret = grads[0];

	const avgGrads = {};
	for (let key in grads[0]) {
		let to_average = []
		for (let grad of grads) {
			// Add a 0th dimension to each gradient, and push it
			// onto a list.
			// Adding a 0th dimension is the same as taking an array
			// and returning [array].
			to_average.push(grad[key].expandDims(0));
		}
		// Concatenate all the expanded gradients into one big tensor, over
		// axis 0. This is basically a tensor containing all the smaller
		// tensors, where the [n]th entry would be grads[n][key].
		let concatenatedTensor = tf.concat(to_average, 0);
		// Take the mean along axis 0, averaging out the gradients.
		avgGrads[key] = tf.mean(concatenatedTensor, 0);
	}
	return avgGrads;

}

async function go() {
	const data = await load();
	const model = await train(data);
	test(model, data);
}

window.go = go


