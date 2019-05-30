package com.lsht.ml;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Junson on 2017/8/22.
 */
public class RnnLayer implements Serializable
{
    public double learning_rate=0.2;
    public  int times =-1;
    public List<double[]> stateList=new ArrayList<>();
    public double[] currentNetValue;
    public double[] currentState;
    public double[] lastState;
    public double[] currentInput;
    public List<double[]> netList=new ArrayList<>();
    public double[][] U;//[state_width][input_width]
    public double[][] W;//[state_width][state_width]
    public double[] bias;
    public double[] gradientBias;
    public double[][] gradientW;
    public double[][] gradientU;
    public int inputWidth;
    public int stateWidth;

    public RnnLayer(int inputWidth, int stateWidth, double learning_rate)
    {
        this.inputWidth=inputWidth;
        this.stateWidth=stateWidth;
        this.learning_rate=learning_rate;

        U = new double[stateWidth][inputWidth]; // # 初始化U

        gradientU = new double[stateWidth][inputWidth];

        W = new double[stateWidth][stateWidth];//  # 初始化W
        CalcOperator.randomUniform(-0.03,0.07,U);
        CalcOperator.randomUniform(-0.03,0.07,W);
        gradientW=new double[stateWidth][stateWidth];
        bias=new double[stateWidth];
        gradientBias=new double[stateWidth];
        CalcOperator.randomUniform(-0.03,0.07,bias);
    }

    /**
     * inputData最后一个恒定为1，表示偏执项
     * @param inputData
     * @return
     */
    public  double[] forward(double[] inputData)
    {
        times ++;

        currentInput=inputData;

        double[] netU=CalcOperator.matrixMultiply(U, inputData);

        double[] netW=null;

        if(times!=0)
        {
            netW=CalcOperator.matrixMultiply(W,
                    currentState);
            currentNetValue = CalcOperator
                    .add(netU, netW);
        }
        else
        {
            currentNetValue = netU;
        }

        // netList.add(currentNetValue);
        lastState=currentState;
        currentState = CalcOperator.softMax(currentNetValue);

        return currentState;
    }

    /**
     * 实现BPTT算法
     *
     * @param label
     */
    public void backward(double[] label)
    {
        double[] sensitivity_array =CalcOperator.calcCECDelta(currentState,
                label);
        calcGradient(sensitivity_array); //计算梯度
    }

    /**
     * 计算公式: TiDu =Delta * Input
     *
     */
    public void calcGradient(double[] sensitivity_array)
    {
        CalcOperator.fill(gradientU,0);

        CalcOperator.fill(gradientW, 0);

        for (int m = 0; m < gradientU.length; m++)
        {
            for (int n = 0; n < gradientU[0].length; n++)
            {
                gradientU[m][n] = sensitivity_array[m] * currentInput[n];
            }
        }

        updateU();


        for (int m = 0; m < gradientW.length; m++)
        {
            for (int n = 0; n < gradientW[0].length; n++)
            {
                gradientW[m][n] = sensitivity_array[m] * lastState[n];
            }
        }

        updateW();

        gradientBias=sensitivity_array;

        updateBias();
    }

    public void updateBias()
    {
        for(int i=0;i<gradientBias.length;i++)
        {
            bias[i]-=gradientBias[i];
        }
    }

    public void updateU()
    {
        // 按照梯度下降，更新权重
        for(int i=0;i<stateWidth;i++)
        {
            for(int j=0;j<inputWidth;j++)
            {
                U[i][j] -=learning_rate * gradientU[i][j];
            }
        }
    }

    public void updateW()
    {
        // 按照梯度下降，更新权重
        for(int i=0;i<W.length;i++)
        {
            for(int j=0;j<W[0].length;j++)
            {
                W[i][j] -=gradientW[i][j]*learning_rate;
            }
        }
    }

    public void reset_state()
    {
        //当前时刻初始化为t0
        times=-1;
        //stateList.clear();
        //netList.clear();
    }

    /**
        每次计算error之前，都要调用reset_state方法重置循环层的内部状态
     */
    private static double error_function(double[] l)
    {
        return CalcOperator.sum(l);
    }

    /**
     *  梯度检查
     */
    public static void gradient_check()
    {
        //0000 001
        //0001 010
        //0010 100
        //0100 001
        //1000 010
        RnnLayer recurrentLayer = new RnnLayer(2,2,0.2);

        double[] x2 = { 0, 1 };
        double[] x3 = { 1, 0 };

        double[] t2 = { 1, 0 };
        double[] t3 = { 0, 1 };


        for (int i = 0; i < 1; i++)
        {
            recurrentLayer.reset_state();
            recurrentLayer.forward(x2);
            recurrentLayer.forward(x3);
            //计算梯度
            recurrentLayer.backward(t3);
        }

        System.out.println("********************************************");

        //检查梯度
        double epsilon = 10e-4;
        for(int i=0;i<recurrentLayer.stateWidth;i++)
        {
            for(int j=0;j<recurrentLayer.stateWidth;j++)
            {
                recurrentLayer.W[i][j]+=epsilon;

                recurrentLayer.reset_state();
                recurrentLayer.forward(x2);
                recurrentLayer.forward(x3);

                double[] y3=recurrentLayer.currentState;
                CalcOperator.print(y3);
                CalcOperator.print(t3);
                double err1 = error_function(CalcOperator.calcCECDelta(y3, t3));

                recurrentLayer.W[i][j]-=2*epsilon;
                recurrentLayer.reset_state();
                recurrentLayer.forward(x2);
                recurrentLayer.forward(x3);
                y3=recurrentLayer.currentState;
                System.out.println("------------------------------------------------------------");
                CalcOperator.print(y3);
                CalcOperator.print(t3);
                double err2 = error_function(CalcOperator.calcCECDelta(y3, t3));

                System.out.println("err1="+err1);
                System.out.println("err2="+err2);

                System.out.println("------------------------------------------------------------");

                double expect_grad = (err1 - err2) / (2.0 * epsilon);
                recurrentLayer.W[i][j] += epsilon;
                System.out.printf("weights(%d,%d): expected  -  actural   %f   -   %f \n",
                    i, j, expect_grad, recurrentLayer.gradientW[i][j]);
                System.out.println(
                        "===================================================================");
            }
        }

    }

    public static void main(String[] args)
    {
        gradient_check();
    }



}
