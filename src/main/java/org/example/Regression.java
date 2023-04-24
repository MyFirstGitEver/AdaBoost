package org.example;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

abstract class PredictFunction {
    abstract float predict(Vector x, Vector w, float b);
    abstract  Vector derivativeByW(Vector w, float b, Pair<Vector, Float>[] dataset);

    public float derivativeByB(Vector w, float b, Pair<Vector, Float>[] dataset) {
        float total = 0;
        int datasetLength = dataset.length;

        for (Pair<Vector, Float> vectorFloatPair : dataset) {
            total += (predict(vectorFloatPair.first, w, b) - vectorFloatPair.second);
        }

        return total / datasetLength;
    }
}

class PolynomialPredictor extends PredictFunction{
    @Override
    public float predict(Vector x, Vector w, float b) {
        return x.dot(w) + b;
    }

    @Override
    public Vector derivativeByW(Vector w, float b, Pair<Vector, Float>[] dataset) {
        Vector derivative = new Vector(w.size());

        int datasetLength = dataset.length;
        int features = w.size();

        for(int i=0;i<features;i++){
            for (Pair<Vector, Float> vectorFloatPair : dataset) {
                float curr = derivative.x(i);

                curr += vectorFloatPair.first.x(i) *
                        (predict(vectorFloatPair.first, w, b) - vectorFloatPair.second);
                derivative.setX(i, curr);
            }

            derivative.setX(i, derivative.x(i) / datasetLength);
        }

        return derivative;
    }
}

// w1*sin(x1) + w0*sin(x0) + b
class SinePredictor extends PredictFunction{
    @Override
    public float predict(Vector x, Vector w, float b) {
        return (float) (w.x(1) * Math.sin(x.x(1)) + w.x(0) * Math.sin(x.x(0)) + b);
    }

    @Override
    public Vector derivativeByW(Vector w, float b, Pair<Vector, Float>[] dataset) {
        Vector derivative = new Vector(w.size());

        int datasetLength = dataset.length;
        int features = w.size();

        for(int i=0;i<features;i++){
            for (Pair<Vector, Float> vectorFloatPair : dataset) {
                float curr = derivative.x(i);

                curr += Math.sin(vectorFloatPair.first.x(i)) * (
                        predict(vectorFloatPair.first, w, b) - vectorFloatPair.second);
                derivative.setX(i, curr);
            }

            derivative.setX(i, derivative.x(i) / datasetLength);
        }

        return derivative;
    }
}

// w = (w2, w1, w0);
// b float
// x = (x2, x1, x0)
class LinearRegression extends Regression {

    LinearRegression(PredictFunction predictor, Pair<Vector, Float>[] dataset) {
        super(predictor, dataset);
    }

    // R-squared error
    @Override
    public float cost(){
        int n = dataset.length;
        float total = 0;

        for(Pair<Vector, Float> p : dataset){
            float term = (p.second - predictor.predict(p.first, w, b));
            total += term * term;
        }

        return total / (2 * n);
    }

    @Override
    public float predict(Vector x){
        if(x.size() != w.size()){
            return Float.NaN;
        }

        return predictor.predict(x, w, b);
    }
}

public abstract class Regression{
    protected final PredictFunction predictor;
    protected final Vector w;
    protected float b;
    protected final Pair<Vector, Float>[] dataset;
    Regression(PredictFunction predictor, Pair<Vector, Float>[] dataset) {
        this.predictor = predictor;
        this.w = new Vector(dataset[0].first.size());
        this.dataset = dataset;
    }

    abstract public float cost();
    abstract protected float predict(Vector x);

    public void train(float learningRate, int iter){
        float cost;
        int iteration = 0;

        while((cost = Math.abs(cost())) > 0.0001 && iteration < iter){
            Vector v = predictor.derivativeByW(w, b, dataset).scaleBy(learningRate);

            b -= learningRate * predictor.derivativeByB(w, b, dataset);
            w.subtract(v);
            iteration++;
        }

        System.out.println(cost);
    }

    private void saveParams() throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter("w.param"));
        for(int i=0;i<w.size();i++){
            writer.write(Float.toString(w.x(i)));
            writer.newLine();
        }

        writer.close();

        writer = new BufferedWriter(new FileWriter("b.param"));
        for(int i=0;i<b;i++){
            writer.write(Float.toString(b));
            writer.newLine();
        }

        writer.close();
    }
}

class LogisticRegression extends Regression implements BinaryClassifier {

    LogisticRegression(PredictFunction predictor, Pair<Vector, Float>[] dataset) {
        super(predictor, dataset);
    }

    @Override
    public float cost(){
        float total = 0;

        // -(yi * log(sigmoid) + (1 - yi)*log(1 - sigmoid))

        for(Pair<Vector, Float> point : dataset){
            float predictPercent = predictor.predict(point.first, w, b);

            total -= (point.second * Math.log(predictPercent + 0.0001))
                    + (1 - point.second) * Math.log(1 - predictPercent + 0.0001);
        }

        return total / dataset.length;
    }

    @Override
    protected float predict(Vector x) {
        return predictor.predict(x, w, b);
    }

    public boolean isPositive(Vector x){
        return predict(x) >= 0.5f;
    }
}