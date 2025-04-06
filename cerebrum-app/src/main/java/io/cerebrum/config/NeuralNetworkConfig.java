package io.cerebrum.config;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class NeuralNetworkConfig {
	
    @Value("${neural-network.inputSize}")
	private long inputSize;
	

    @Value("${neural-network.outputSize}")
	private long outputSize;
	
	

	@Bean
	public MultiLayerConfiguration createConfiguration() {
		return new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).updater(new Adam(0.001)).list()
				.layer(new DenseLayer.Builder().nIn(inputSize).nOut(64).activation(Activation.RELU).build())
				.layer(new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.RELU).build())
				.layer(new OutputLayer.Builder().nIn(32).nOut(outputSize).activation(Activation.SOFTMAX)
						.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build())
				.build();
	}
}
