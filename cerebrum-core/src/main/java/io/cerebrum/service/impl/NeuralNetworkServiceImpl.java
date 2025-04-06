package io.cerebrum.service.impl;

import java.util.List;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import io.cerebrum.Feature;
import io.cerebrum.TrainingData;
import io.cerebrum.service.NeuralNetworkService;

@Service
public class NeuralNetworkServiceImpl implements NeuralNetworkService, InitializingBean {

	private static Logger log = LoggerFactory.getLogger(NeuralNetworkServiceImpl.class);

	private MultiLayerNetwork model;

	@Autowired
	private MultiLayerConfiguration config;

	public void train(TrainingData trainingData) {
		int inputSize = trainingData.getInputs().get(0).getFeatures().size();
		int outputSize = trainingData.getLabels().stream().max(Integer::compare).orElse(0) + 1;


		INDArray features = Nd4j.create(trainingData.getInputs().size(), inputSize);
		INDArray labels = Nd4j.zeros(trainingData.getLabels().size(), outputSize);

		for (int i = 0; i < trainingData.getInputs().size(); i++) {
			List<Double> row = trainingData.getInputs().get(i).getFeatures();
			for (int j = 0; j < inputSize; j++) {
				features.putScalar(new int[] { i, j }, row.get(j));
			}
			labels.putScalar(new int[] { i, trainingData.getLabels().get(i) }, 1.0);
		}

		model.fit(features, labels);
		log.info("Training complete");
	}

	public int predict(Feature input) {
		INDArray inputVec = Nd4j.create(input.getFeatures());
		INDArray output = model.output(inputVec);
		return Nd4j.argMax(output, 1).getInt(0);
	}

	@Override
	public void afterPropertiesSet() throws Exception {
		model = new MultiLayerNetwork(config);
		model.init();
	}

}
