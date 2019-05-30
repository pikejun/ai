package com.lsht.ml;

/**
 * Created by Junson on 2017/8/20.
 */
public class LinearUnit extends  Perceptron
{
    private double a=1;
    private double c=0;

    public LinearUnit(int input_num)
    {
        init(input_num, this::linearFunc);
    }

    private double linearFunc(double x)
    {
        return a*x+c;
    }

    public double getA() {
        return a;
    }

    public void setA(double a) {
        this.a = a;
    }

    public double getC() {
        return c;
    }

    public void setC(double c) {
        this.c = c;
    }

    public static void main(String[] args)
    {
       // # 输入向量列表，每一项是工作年限
       double[][] input_vecs = {{5}, {3}, {8}, {1.4}, {10.1}};
      //  # 期望的输出列表，月薪，注意要与输入一一对应
       double[] labels = {5500, 2300, 7600, 1800, 11400};

        LinearUnit linearUnit=new LinearUnit(1);

        linearUnit.train(input_vecs,labels,10,0.01);

        System.out.println(linearUnit.predict(new double[]{10}));

    }
}
