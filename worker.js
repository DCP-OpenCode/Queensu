/**
 * @file    worker.js
 *          Used for training over the DCP network.
 *          Adapted from the tensorflow.js mnist-core example.
 * @author  Ian Chew
 * @date    Jan 2019
 */

/**
 * Function that's sent to the workers.
 *
 * @param {Object} input_data - The data we feed into the model
 * @param {Array} input_data.images - The images we're training on.
 * @param {Array} input_data.labels - The actual classes each image belongs to.
 * @param {Array} weights_arr - The weights used in the model.
 * @return {Array} - The gradients output by the model when run on the input images and labels.
 */
export async function worker(input_data, weights_arr) {


	tf = require('tf.min.js');

	// MnistModel class gets inserted here.

	/**
	 * Decode the image array sent from the server into a tf.Tensor.
	 *
	 * @param {Array} imageArray - The images sent from the server in input_data.images.
	 * @return {tf.Tensor} - A tensor we can pass to MnistModel.compute().
	 */
	function decode_images(imageArray) {
		// The number of images sent.
		const imageCount = imageArray.length/model.IMAGE_SIZE/model.IMAGE_SIZE;
		// compute() reshapes this tensor to 4D so we don't have to here.
		return tf.tensor(imageArray, [imageCount, model.IMAGE_SIZE*model.IMAGE_SIZE]);
	}

	/**
	 * Decode the label array sent from the server into a tf.Tensor.
	 *
	 * @param {Array} imageArray - The images sent from the server in input_data.images.
	 * @return {tf.Tensor} - A tensor we can pass to MnistModel.getGradients().
	 */
	function decode_labels(labels) {
		const labelCount = labels.length/model.LABELS_SIZE;
		return tf.tensor(labels, [labelCount, model.LABELS_SIZE]);
	}

	// Convert the input into tensors.
	const model = new MnistModel(weights_arr);
	const images = decode_images(input_data.images);
	const labels = decode_labels(input_data.labels);

	// We're about 25% complete.
	progress(0.25);

	// Compute the gradients, running the images through the network.
	const grads = await model.getGradients(images, labels);

	// Encode the gradients and return them over the network.
	const networkGrads = model.encode(grads);

	// Clean up memory used by the model.
	model.dispose();
	tf.dispose(grads);
	tf.dispose(images);
	tf.dispose(labels);

	// We're 100% complete.
	progress(1.00);
	return networkGrads
} // End worker
