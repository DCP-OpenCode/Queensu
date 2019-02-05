/**
 * @file    mnist.js
 *          Used for training over the DCP network.
 *          Adapted from the tensorflow.js mnist-core example.
 * @author  Ian Chew
 * @date    Jan 2019
 */

/**
 * Class containing the model's weights, and methods to use them.
 *
 */
export class MnistModel {
	constructor(weightsArr) {
		// Constants which you can configure:

		// Hyperparameters.
		this.LEARNING_RATE = .1;
		this.BATCH_SIZE = 64;
		this.TRAIN_STEPS = 100;

		// Data constants.
		this.IMAGE_SIZE = 28;
		this.LABELS_SIZE = 10;
		this.optimizer = tf.train.sgd(this.LEARNING_RATE);

		// Number of filters each layer outputs.
		this.conv1OutputDepth = 8;
		this.conv2InputDepth = this.conv1OutputDepth;
		this.conv2OutputDepth = 16;
		// End constants.

		if (typeof weightsArr === "undefined") {
			// Initialize weights randomly if we're not given weights (for client).
			this.conv1Weights =
			    tf.variable(tf.randomNormal([5, 5, 1, this.conv1OutputDepth], 0, 0.1));
			this.conv2Weights = tf.variable(
			    tf.randomNormal([5, 5, this.conv2InputDepth, this.conv2OutputDepth], 0, 0.1));
			this.fullyConnectedWeights = tf.variable(tf.randomNormal(
			    [7 * 7 * this.conv2OutputDepth, this.LABELS_SIZE], 0,
			    1 / Math.sqrt(7 * 7 * this.conv2OutputDepth)));
			this.fullyConnectedBias = tf.variable(tf.zeros([this.LABELS_SIZE]));

			// Use this for iteration later.
			this.layers = [
			    this.conv1Weights,
			    this.conv2Weights,
			    this.fullyConnectedWeights,
			    this.fullyConnectedBias
			];
		} else {
			// Initialize weights based on input (for worker).
			// This is because tensors cannot currently be directly sent over the network.

			this.layers = this.decode(weightsArr);

			// Give each element of this.layers a nicer name.
			[this.conv1Weights,
			 this.conv2Weights,
			 this.fullyConnectedWeights,
			 this.fullyConnectedBias] = this.layers;
		}
	}

	/**
	 * Decode the weights (or gradients) that were sent over the network into tensors.
	 *
	 * @param {Array<tf.Tensor>} weightsArr - The array of weights sent over the network.
	 * @returns {Array<Array<number>>} - The weights, encoded as an array of tensors. Each
	 * element of the array is one layer.
	 */
	decode(weightsArr) {
		const tensors = [
			tf.tensor(weightsArr[0], [5, 5, 1, this.conv1OutputDepth]),
			tf.tensor(weightsArr[1], [5, 5, this.conv2InputDepth, this.conv2OutputDepth]),
			tf.tensor(weightsArr[2], [7 * 7 * this.conv2OutputDepth, this.LABELS_SIZE]),
			tf.tensor(weightsArr[3])
		];

		const vars = []
		for (let tensor of tensors) {
			vars.push(tf.variable(tensor));
			// Clean up the tensor:
			tensor.dispose();
		}

		return vars;
	}

	/** Converts the weights (or gradients) to an array to be sent over the network.
	 * You should be able to pass this array to the constructor to get back the
	 * original model.
	 *
	 * @param {Array<tf.Tensor>} - The weights, encoded as an array of tensors.
	 * @return {Array<Array<number>>} - An array containing the weights of the network.
	 */
	encode(tensorArr) {
		const ret = [];
		for (let key in tensorArr) {
			ret.push(Array.from(tensorArr[key].dataSync()));
		}
		return ret;
	}

	/**
	 * Converts the model into an format that can be sent over the network.
	 */
	toArray() {
		return this.encode(this.layers);
	}

	/** Run the model on the inputs.
	 *
	 * @param {tf.Tensor} inputXs - A '4D' tensor, containing a batch of images.
	 * @return {tf.Tensor} - A 2D tensor containing the category values for each image.
	 */
	compute(inputXs) {
		const xs = inputXs.as4D(-1, this.IMAGE_SIZE, this.IMAGE_SIZE, 1);

		const strides = 2;
		const pad = 0;

		// Conv 1
		const layer1 = tf.tidy(() => {
			return xs.conv2d(this.conv1Weights, 1, 'same')
					.relu()
					.maxPool([2, 2], strides, pad);
		});

		// Conv 2
		const layer2 = tf.tidy(() => {
			return layer1.conv2d(this.conv2Weights, 1, 'same')
					.relu()
					.maxPool([2, 2], strides, pad);
		});

		// Final layer
		return layer2.as2D(-1, this.fullyConnectedWeights.shape[0])
			.matMul(this.fullyConnectedWeights)
			.add(this.fullyConnectedBias);
	}
	
	/**
	 * Compute the loss between the output from compute() and
	 * the labels.
	 *
	 * @param {tf.Tensor} labels - The ground truth, the expected results.
	 * @param {tf.Tensor} ys - The results produced by the model using compute().
	 * @result {tf.Tensor}
	 */
	loss(labels, ys) {
		return tf.losses.softmaxCrossEntropy(labels, ys).mean();
	}

	/**
	 * Compute gradients given a set of images and labels.
	 *
	 * @param {tf.Tensor} xs - A 2D tensor containing the input images.
	 * @param {tf.Tensor} labels - A 2D tensor containing the input labels.
	 * @result {tf.Tensor}
	 */
	async getGradients(xs, labels) {
//		const returnCost = true;
//		for (let i = 0; i < TRAIN_STEPS; i++) {
			// Compute the gradients
			const {cost, grads} = this.optimizer.computeGradients(() => {
				return this.loss(labels, this.compute(xs));
			});
			// Free up cost tensor.
//			cost.dispose();
//			log(cost.dataSync(), i);
//		}
		return grads;
	}

	/**
	 * Apply gradients to the internal model.
	 *
	 * @param {Object} grads - The gradients produced from getGradients.
	 */
	applyGradients(grads) {
		this.optimizer.applyGradients(grads);
	}

	/**
	 * Predict the digit number from a batch of input images.
	 *
	 * @param {tf.Tensor} xs - The batch of input images.
	 * @returne {Array} - An array containing the predicted values.
	 */
	predict(xs) {
		const pred = tf.tidy(() => {
			const axis = 1;
			return this.compute(xs).argMax(axis);
		});

		const ret = Array.from(pred.dataSync());
		// Clean up the tensor.
		pred.dispose();
		return ret;
	}

	/**
	 * Given a logits or label vector, return the class indices.
	 *
	 * @param {tf.Tensor} labels - The labels as a 2D tensor.
	 * @return {Array} - An 1D array containing the correct label from
	 * 0 to 9.
	 */
	classesFromLabel(labels) {
		const axis = 1;
		// Get the maximum index of each row of the 2D tensor.
		const indices = labels.argMax(axis);

		const ret = Array.from(indices.dataSync());
		// Clean up the tensor.
		indices.dispose();
		return ret;
	}

	/**
	 * Clean up the memory used by the model's tensors.
	 */
	dispose() {
		this.layers.forEach(layer => layer.dispose());
		this.optimizer.dispose();
	}

}

