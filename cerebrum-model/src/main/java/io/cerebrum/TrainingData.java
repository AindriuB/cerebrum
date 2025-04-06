package io.cerebrum;

import java.util.List;

public class TrainingData {
	
    private List<Feature> inputs;
	private List<Integer> labels;

	public List<Feature> getInputs() {
		return inputs;
	}

	public void setInputs(List<Feature> inputs) {
		this.inputs = inputs;
	}

	public List<Integer> getLabels() {
		return labels;
	}

	public void setLabels(List<Integer> labels) {
		this.labels = labels;
	}

}
