package weka.filters.supervised.attribute;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

public class urCAIM extends Filter implements SupervisedFilter, OptionHandler
{
	private static final long serialVersionUID = 1L;

	protected Range m_DiscretizeCols = new Range();
	protected boolean m_OutputInNumeric = false;
	protected ArrayList<ArrayList<Double>> SchemeList;
	protected int classAttributeIndex = -1;
	protected int numberClasses = -1;
	protected Instances dataset;

	public urCAIM() {
		setAttributeIndices("first-last");
	}

	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>();

		newVector.addElement(new Option("\tSpecifies list of columns to Discretize. First"
				+ " and last are valid es.\n" + "\t(default none)",
				"R", 1, "-R <col1,col2-col4,...>"));

		newVector.addElement(new Option("\tOutput in numeric format.", "O", 1,"-O"));

		return newVector.elements();
	}

	public void setOptions(String[] options) throws Exception {
		setOutputInNumeric(Utils.getFlag('O', options));

		String convertList = Utils.getOption('R', options);

		if (convertList.length() != 0)
			setAttributeIndices(convertList);
		else
			setAttributeIndices("first-last");

		if (getInputFormat() != null)
			setInputFormat(getInputFormat());
	}

	public String[] getOptions() {

		String[] options = new String[20];
		int current = 0;

		options[current++] = "-O";
		options[current++] = "" + m_OutputInNumeric;

		if (!getAttributeIndices().equals(""))
			options[current++] = "-R";

		options[current++] = getAttributeIndices();

		while (current < options.length)
			options[current++] = "";

		return options;
	}

	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		result.enableAllAttributes();
		result.enable(Capability.UNARY_CLASS);
		result.enable(Capability.NOMINAL_CLASS);
		return result;
	}

	public boolean setInputFormat(Instances instanceInfo) throws Exception {
		super.setInputFormat(instanceInfo);
		m_DiscretizeCols.setUpper(instanceInfo.numAttributes() - 1);
		return false;
	}

	public boolean input(Instance instance) {
		if (getInputFormat() == null)
			throw new IllegalStateException("No input instance format defined");
		bufferInput(instance);
		return false;
	}

	protected void urCAIMsplit(int attribute)
	{
		Vector<Double> temp = new Vector<Double>();

		for (int i = 0; i < dataset.numInstances(); i++)
			if (temp.indexOf(dataset.instance(i).value(attribute)) < 0)
				temp.add(dataset.instance(i).value(attribute));

		Object[] instancesValues = temp.toArray();
		Arrays.sort(instancesValues);

		int[][] appearanceMatrix = new int[numberClasses][instancesValues.length];

		for (int i = 0; i < dataset.numInstances(); i++)
		{
			int row = (int) dataset.instance(i).classValue();
			int col = Arrays.binarySearch(instancesValues, dataset.instance(i).value(attribute));

			appearanceMatrix[row][col]++;
		}

		Vector<Double> midPoints = new Vector<Double>();
		ArrayList<Integer> D = new ArrayList<Integer>();

		midPoints.add(Double.parseDouble(instancesValues[0].toString()));

		for (int i = 0; i < instancesValues.length - 1; i++)
			midPoints.add((Double.parseDouble(instancesValues[i].toString()) + Double.parseDouble(instancesValues[i + 1].toString())) / 2);

		midPoints.add(Double.parseDouble(instancesValues[instancesValues.length - 1].toString()));

		D.add(0);
		D.add(midPoints.size()-1);

		int[] numberInstancesClass = new int[numberClasses];

		for(int j = 0; j < dataset.numInstances(); j++)
			numberInstancesClass[(int) dataset.instance(j).classValue()]++;

		double maxCAIM = 0;

		while (true)
		{
			int bestmidPoint = -1;

			for (int i = 1; i < midPoints.size() - 1; i++)
			{
				double CAIMvalue = computeFitness(attribute, i, D, dataset.numInstances(), numberInstancesClass, appearanceMatrix);

				if(CAIMvalue > maxCAIM)
				{
					maxCAIM = CAIMvalue;
					bestmidPoint = i;
				}
			}

			if(bestmidPoint == -1)
				break;
			else
			{
				for(int i = 0; i < D.size(); i++)
					if(D.get(i) > bestmidPoint)
					{
						D.add(i, bestmidPoint);
						break;
					}
			}
		}

		ArrayList<Double> array = new ArrayList<Double>();

		for(int i = 0; i < D.size(); i++)
			array.add(midPoints.get(D.get(i)));

		SchemeList.set(attribute, array);
	}

	private double computeFitness(int attribute, int point, ArrayList<Integer> D, int numberInstances, int[] numberInstancesClass, int[][] quantaMatrix)
	{
		ArrayList<Integer> newD = new ArrayList<Integer>(D);

		for(int i = 0; i < newD.size(); i++)
			if(newD.get(i) > point)
			{
				newD.add(i, point);
				break;
			}

		int[][] newQuantaMatrix = buildQuantaMatrix(quantaMatrix, newD);

		int numberIntervals = newQuantaMatrix[0].length;
		long[] maxValues = new long[numberIntervals];
		long[] sumValues = new long[numberIntervals];

		for(int j = 0; j < numberIntervals; j++)
		{
			int sum = 0;
			int max = -1;

			for(int k = 0; k < numberClasses; k++)
			{
				sum += newQuantaMatrix[k][j];

				if(newQuantaMatrix[k][j] > max)
					max = newQuantaMatrix[k][j];
			}

			maxValues[j] = max;
			sumValues[j] = sum;
		}

		double INFO = 0, I = 0, H = 0;

		for(int k = 0; k < numberClasses; k++)
			for(int j = 0; j < numberIntervals; j++)
			{
				double Pir = newQuantaMatrix[k][j] / (double) numberInstances;
				double Pi = numberInstancesClass[k] / (double) numberInstances;
				double Pr = sumValues[j] / (double) numberInstances;

				if(Pir != 0)
				{
					H += Pir * Math.log10(1.0 / Pir) / Math.log(2);
					I += (1-Pi) * Pir * Math.log10(Pir / (Pi * Pr)) / Math.log(2);
					INFO += Pir * Math.log10(Pr / Pir) / Math.log(2);
				}
			}

		double[] results = new double[3];
		double maxCAIM = 0;

		for(int j = 0; j < numberIntervals; j++)
		{
			results[0] += (maxValues[j] * maxValues[j]) / (double) sumValues[j];
			maxCAIM += sumValues[j];
		}

		results[0] /= numberIntervals;							// CAIM
		results[0] /= maxCAIM;									// CAIM Normalized
		results[1] = I / H;										// CAIR
		results[2] = 1.0 - (INFO / H);							// 1 - CAIU

		return results[0] * results[1] * results[2];
	}

	private int[][] buildQuantaMatrix(int[][] quantaMatrix, ArrayList<Integer> newD)
	{
		int[][] newQuantaMatrix = new int[numberClasses][newD.size()-1];

		for(int i = 0; i < numberClasses; i++)
		{
			for(int j = 0; j < newD.size()-1; j++)
			{
				for(int k = newD.get(j); k < newD.get(j+1); k++)
				{
					newQuantaMatrix[i][j] += quantaMatrix[i][k];
				}
			}
		}

		return newQuantaMatrix;
	}

	public boolean batchFinished() throws Exception
	{
		dataset = getInputFormat();

		if (dataset == null)
			throw new IllegalStateException("No input instance format defined");

		SchemeList = new ArrayList<>();

		Attribute classAttribute = dataset.attribute(dataset.classIndex());

		if (classAttribute == null)
			throw new Exception("Wrong name in class's attribute");

		classAttributeIndex = classAttribute.index();

		numberClasses = classAttribute.numValues();

		for (int current = 0; current < dataset.numAttributes(); current++)
			SchemeList.add(new ArrayList<Double>());

		ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

		for (int current = 0; current < dataset.numAttributes()-1; current++)
		{
			if (!dataset.attribute(current).isNumeric())
				continue;

			threadExecutor.execute(new evaluationThread(current));
		}

		threadExecutor.shutdown();

		try
		{
			if (!threadExecutor.awaitTermination(30, TimeUnit.DAYS))
				System.out.println("Threadpool timeout occurred");
		}
		catch (InterruptedException ie)
		{
			System.out.println("Threadpool prematurely terminated due to interruption in thread that created pool");
		}

		setOutputFormat();

		for (int i = 0; i < getInputFormat().numInstances(); i++)
			convertInstance(getInputFormat().instance(i));

		return true;
	}

	public String globalInfo() {
		return "An instance filter that discretizes a range of numeric";
	}

	public String outputInNumericTipText() {
		return "true:output in 1,2,3, false:output in [a,b),[c,d), format";
	}

	public boolean getOutputInNumeric() {
		return m_OutputInNumeric;
	}

	public void setOutputInNumeric(boolean val) {
		m_OutputInNumeric = val;
	}

	public String factorTipText() {
		return "if not automate bins, input number of bins";
	}

	public String metricTipText() {
		return "Select metric type";
	}

	public String attributeIndicesTipText() {
		return "Specify range of attributes to act on."
				+ " This is a comma separated list of attribute indices, with"
				+ " \"first\" and \"last\" valid values. Specify an inclusive"
				+ " range with \"-\". E.g: \"first-3,5,6-10,last\".";
	}

	public String getAttributeIndices() {
		return m_DiscretizeCols.getRanges();
	}

	public void setAttributeIndices(String rangeList) {
		m_DiscretizeCols.setRanges(rangeList);
	}

	public void setAttributeIndicesArray(int[] attributes) {
		setAttributeIndices(Range.indicesToRangeList(attributes));
	}

	protected void setOutputFormat()
	{
		Instances Data = getInputFormat();
		ArrayList<Attribute> attributes = new ArrayList<Attribute>(Data.numAttributes());

		for (int current = 0; current < Data.numAttributes(); current++)
		{
			if (!Data.attribute(current).isNumeric() || !m_DiscretizeCols.isInRange(current) || current == classAttributeIndex)
			{
				attributes.add((Attribute) Data.attribute(current).copy());
				continue;
			}

			ArrayList<String> attValues = new ArrayList<String>();
			ArrayList<Double> l = SchemeList.get(current);

			for (int i = 0; i < l.size() - 1; i++)
			{
				if (m_OutputInNumeric)
					attValues.add(Integer.toString(i));
				else
				{
					String s = "[" + l.get(i).toString() + "-" + l.get(i + 1).toString() + ")";
					if (i == (l.size() - 2))
						s = s.replace(")", "]");

					attValues.add(s);
				}
			}

			attributes.add(new Attribute(Data.attribute(current).name(),	attValues));

		}

		Instances outputFormat = new Instances(Data.relationName(), attributes, 0);
		setOutputFormat(outputFormat);
	}

	protected void convertInstance(Instance instance)
	{
		double[] vals = new double[outputFormatPeek().numAttributes()];

		Instances Data = getInputFormat();

		for (int current = 0; current < Data.numAttributes(); current++)
		{
			if (!Data.attribute(current).isNumeric() || !m_DiscretizeCols.isInRange(current) || current == classAttributeIndex)
				vals[current] = instance.value(current);
			else
			{
				ArrayList<Double> l = SchemeList.get(current);

				int k = 0;
				while (instance.value(current) > Double.parseDouble(l.get(k).toString()))
				{
					k++;
					if (k == l.size())
						break;
				}
				k--;
				if (k < 0)		k = 0;
				if(k == l.size()-1)	k--;
				vals[current] = k;
			}
		}

		Instance inst = null;

		if (instance instanceof SparseInstance) {
			inst = new SparseInstance(instance.weight(), vals);
		} else {
			inst = new DenseInstance(instance.weight(), vals);
		}

		inst.setDataset(getOutputFormat());
		copyValues(inst, false, instance.dataset(), getOutputFormat());
		inst.setDataset(getOutputFormat());
		push(inst);
	}

	protected Instance convertInstanceTest(Instance instance)
	{
		double[] vals = new double[outputFormatPeek().numAttributes()];

		Instances Data = getInputFormat();

		for (int current = 0; current < Data.numAttributes(); current++)
		{
			if (!Data.attribute(current).isNumeric() || !m_DiscretizeCols.isInRange(current) || current == classAttributeIndex)
				vals[current] = instance.value(current);
			else
			{
				ArrayList<Double> l = SchemeList.get(current);

				int k = 0;
				while (instance.value(current) > Double.parseDouble(l.get(k).toString()))
				{
					k++;
					if (k == l.size())
						break;
				}
				k--;
				if (k < 0)		k = 0;
				if(k == l.size()-1)	k--;
				vals[current] = k;
			}
		}

		Instance inst = null;

		if (instance instanceof SparseInstance) {
			inst = new SparseInstance(instance.weight(), vals);
		} else {
			inst = new DenseInstance(instance.weight(), vals);
		}

		return inst;
	}

	/////////////////////////////////////////////////////////////////
	//-------------------------------------------- Evaluation Thread
	/////////////////////////////////////////////////////////////////

	private class evaluationThread extends Thread
	{
		private int attribute;

		public evaluationThread(int attribute)
		{
			this.attribute = attribute;
		}

		public void run()
		{
			urCAIMsplit(attribute);
		}
	}
}