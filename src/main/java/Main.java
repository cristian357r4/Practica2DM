import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

public class Main {
    public static void main(String[] args) throws Exception {
//load data
        Instances data = new Instances(new BufferedReader(new
                FileReader("dataset/linear.arff")));
        data.setClassIndex(data.numAttributes() - 1);
//build model
        LinearRegression model = new LinearRegression();
        model.buildClassifier(data); //the last instance with missing class is not used
        System.out.println(model);

//classify the last instance
        Instance myHouse = data.lastInstance();
        Instance test = new DenseInstance(6);
        //estos valores se llenan con los jtexfield
        test.setValue(0, 800);
        test.setValue(1, 3000);
        test.setValue(2, 6);
        test.setValue(3, 1);
        test.setValue(4, 1);
        double price = model.classifyInstance(myHouse);
        double price1 = model.classifyInstance(test);
        System.out.println("My house (" + myHouse + "): " + price);
    }
}