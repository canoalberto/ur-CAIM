package runnable;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;

public class urCAIM
{
	public static void main(String[] args) throws Exception {
		
		// Load dataset
		BufferedReader reader = new BufferedReader(new FileReader("data/iris.arff"));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		data.setClassIndex(data.numAttributes() - 1);
		
		// Discretize data
		Filter filter = new weka.filters.supervised.attribute.urCAIM();
		filter.setInputFormat(data);
		Instances discretized = Filter.useFilter(data, filter);
		
		// Save discretized dataset
	    ArffSaver arffSaver = new ArffSaver();
	    arffSaver.setInstances(discretized);
	    arffSaver.setFile(new File("data/iris-discretized.arff"));
	    arffSaver.writeBatch();
	}
}