package org.example;

import java.awt.*;

public class Vector {
    private float[] points;

    public Vector(float... points) {
        this.points = points;
    }

    public Vector(int size) {
        points = new float[size];
    }

    public float x(int i) {
        return points[i];
    }

    public void setX(int pos, float value) {
        points[pos] = value;
    }

    public int size() {
        return points.length;
    }

    public float distanceFrom(Vector x) {
        if (size() != x.size()) {
            return Float.NaN;
        }

        float total = 0;
        for (int i = 0; i < x.size(); i++) {
            total += (x.x(i) - x(i)) * (x.x(i) - x(i));
        }

        return (float) Math.sqrt(total);
    }

    public void add(Vector v) {
        for (int i = 0; i < points.length; i++) {
            points[i] += v.x(i);
        }
    }

    public Vector scaleBy(float x) {
        for (int i = 0; i < points.length; i++) {
            points[i] *= x;
        }

        return this;
    }

    public int intRGB() {
        Color color = new Color(Math.round(points[0]), Math.round(points[1]), Math.round(points[2]));

        return color.getRGB();
    }

    public Vector iterativeMul(Vector v) {
        if (points.length != v.size()) {
            return null;
        }

        Vector ans = new Vector(v.size());

        for (int i = 0; i < points.length; i++) {
            ans.setX(i, v.x(i) * points[i]);
        }

        return ans;
    }

    public void concat(Vector v) {
        float[] newVec = new float[size() + v.size()];

        System.arraycopy(points, 0, newVec, 0, points.length);

        for (int i = points.length; i < newVec.length; i++) {
            newVec[i] = v.x(i - points.length);
        }

        points = newVec;
    }

    public void normalise() {
        float length = 0.0f;

        for (int i = 0; i < points.length; i++) {
            length += points[i] * points[i];
        }

        length = (float) Math.sqrt(length);

        if (length == 0) {
            return;
        }

        for (int i = 0; i < points.length; i++) {
            points[i] /= length;
        }
    }

    float dot(Vector w){
        if(points.length != w.size()){
            return Float.NaN;
        }

        int n = points.length;
        float total = 0;
        for(int i=0;i<n;i++){
            total += points[i] * w.x(i);
        }

        return total;
    }

    void subtract(Vector v){
        for(int i=0;i<points.length;i++){
            points[i] -= v.x(i);
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        builder.append("(");
        for (float point : points) {
            builder.append(point);
            builder.append(", ");
        }

        return builder.append(")").toString();
    }
}