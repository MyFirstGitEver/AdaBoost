package org.example;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

// --- TRained by AdaBoost
//        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\train.xlsx");
//
//        Pair<Vector, Float>[] dataset = new Pair[reader.getRowCount() - 1];
//
//        for (int i = 0; i < dataset.length; i++) {
//            Object[] data;
//            try {
//                data = reader.getRow(i + 1, 0);
//            } catch (Exception e) {
//                break;
//            }
//
//            double label = (double) data[0];
//            if(label == 0){
//                label = -1;
//            }
//
//            float em;
//
//            if (data[7] == null) {
//                em = 0;
//            } else if (data[7].equals("S")) {
//                em = 1;
//            } else {
//                em = 2;
//            }
//            Vector v = new Vector(
//                    ((Double) data[1]).floatValue(),
//                    data[2].equals("male") ? 0 : 1,
//                    data[3] == null ? 10 : ((Double) data[3]).floatValue(),
//                    ((Double) data[4]).floatValue(),
//                    ((Double) data[5]).floatValue(),
//                    ((Double) data[6]).floatValue(),
//                    em);
//
//            dataset[i] = new Pair<>(v, (float) label);
//        }
//
//        AdaBoost model = new AdaBoost(dataset);
//
//        model.train(30);
//        test(model);

// --- Trained by LogisticModel
//ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\train.xlsx");
//        Pair<Vector, Float>[] dataset = reader.createLabeledDataset(0, 0, 0, 0, 1);
//
//        //Pair<Vector, Float>[] trainData = Arrays.copyOfRange(dataset, 0, 3000);
//        //Pair<Vector, Float>[] testData = Arrays.copyOfRange(dataset, 3000, dataset.length);
//
//        LogisticRegression model = new LogisticRegression(new LogisticPredictor(), dataset);
//
//        model.train(0.001f, 90_000);
//        test(model, new ExcelReader("D:\\Source code\\Data\\Titanic\\test.xlsx").createLabeledDataset(0, 0, 0, 0, 1));

//        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Stroke\\stroke.xlsx");
//                Pair<Vector, Float>[] dataset = reader.createLabeledDataset(0, 0, 0, -1, 1);
//
//        Pair<Vector, Float>[] trainData = Arrays.copyOfRange(dataset, 0, 3000);
//        Pair<Vector, Float>[] testData = Arrays.copyOfRange(dataset, 3000, dataset.length);
//
//        LogisticRegression model = new LogisticRegression(new LogisticPredictor(), trainData);
//
//        model.train(0.001f, 5_000);
//        test(model, testData);

public class Main {
    public static void main(String[] args) throws Exception {
        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Stroke\\stroke.xlsx");
                Pair<Vector, Float>[] dataset = reader.createLabeledDataset(0, 0, 0, -1, 1);

        Pair<Vector, Float>[] trainData = Arrays.copyOfRange(dataset, 0, 3000);
        Pair<Vector, Float>[] testData = Arrays.copyOfRange(dataset, 3000, dataset.length);

        AdaBoost model = new AdaBoost(trainData);
        model.train(100);
        test(model, testData);
    }

    static void test(BinaryClassifier classifier, Pair<Vector, Float>[] dataset) throws Exception {

        int hit = 0;
        for(Pair<Vector, Float> p : dataset){
            boolean survived = classifier.isPositive(p.first);

            if((survived && p.second == 1) || (!survived && p.second == 0)){
                hit++;
            }
        }
        System.out.println(hit / (float)dataset.length * 100 + "%");
    }

    private float[] stringsToNums(String[] data){
        HashMap<String, Integer> set = new HashMap<>();
        float[] nums = new float[data.length];

        int index = 0;
        for(int i=0;i<data.length;i++){
            String point = data[i];
            if(set.get(point) == null){
                set.put(point, index);
                index++;
            }

            nums[i] = set.get(point);
        }

        return nums;
    }
}

class LogisticPredictor extends PolynomialPredictor{
    @Override
    public float predict(Vector x, Vector w, float b) {
        float term = x.dot(w) + b;

        return (float) (1 / (Math.exp(-term) + 1));
    }
}