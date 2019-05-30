package com.lsht.ml;

import java.io.Serializable;
import java.util.*;
/**
 * Created by Junson on 2017/8/22.
 */
public class RecurrentLayer implements Serializable
{
    public double learning_rate=0.1;
    public  int times =-1;
    public List<double[]> stateList=new ArrayList<>();
    public double[] lastState;
    public double[] currentState;
    public List<double[]> netList=new ArrayList<>();
    public double[][] U;//[state_width][input_width]
    public double[][] W;//[state_width][state_width]
    public double biasU;
    public double biasW;


    public List<double[]> deltaUList=new ArrayList<>();
    public List<double[]> deltaWList=new ArrayList<>();

    public List<double[]> inputList=new ArrayList<>();
    public double[][] gradientW;
    public double[][] gradientU;
    public int inputWidth;
    public int stateWidth;


    public double[] getLastState()
    {
       return lastState;
    }

    public  RecurrentLayer(int inputWidth,int stateWidth)
    {
        this.inputWidth=inputWidth;
        this.stateWidth=stateWidth;
        this.learning_rate=0;

        U = new double[stateWidth][inputWidth]; // # 初始化U

        gradientU = new double[stateWidth][inputWidth];

        W = new double[stateWidth][stateWidth];//  # 初始化W
        CalcOperator.randomUniform(1.0/(stateWidth*stateWidth),1.0/(stateWidth*stateWidth),U);
        CalcOperator.randomUniform(1.0/(stateWidth*inputWidth),1.0/(stateWidth*inputWidth),W);
        gradientW=new double[stateWidth][stateWidth];
    }



    public  RecurrentLayer(int inputWidth,int stateWidth,double learning_rate)
    {
        this.inputWidth=inputWidth;
        this.stateWidth=stateWidth;
        this.learning_rate=learning_rate;

        U = new double[stateWidth][inputWidth]; // # 初始化U

        gradientU = new double[stateWidth][inputWidth];

        W = new double[stateWidth][stateWidth];//  # 初始化W
        CalcOperator.randomUniform(-0.0003d,0.0007d,U);
        CalcOperator.randomUniform(-0.0003d, 0.0007d, W);
        gradientW=new double[stateWidth][stateWidth];
    }

    /**
     * 要求输入数据已进行one-hot编码
     * @param inputData
     * @return
     */
    public  double[] forward(double[] inputData)
    {
        times ++;
        lastState = currentState;
        inputList.add(inputData);
        double[] netU=CalcOperator.matrixMultiply(U, inputData);

        double[] netW=null;

        if(times!=0)
        {
            netW=CalcOperator.matrixMultiply(W,
                   CalcOperator.relu(currentState));
            currentState = CalcOperator
                    .add(netU, netW);
        }
        else
        {
            currentState = netU;
        }

        netList.add( CalcOperator.relu(currentState));
        currentState = CalcOperator.softMax(currentState);
        stateList.add(currentState);

        return currentState;
    }

    /**
     * 实现BPTT算法
     *
     * @param label
     */
    public void backward(double[] label)
    {
        double[] sensitivity_array =CalcOperator.softMax(CalcOperator.calcCECDelta(currentState,label));
        calcDelta(sensitivity_array);
        calcGradient(sensitivity_array); //计算梯度
    }

    /**
     * 对one-hot编码的误差数据其误差为： -*logY
     * @param sensitivity_array
     */
    public void calcDelta(double[] sensitivity_array)
    {
        //公式：delta(K)=delta(T)*(W*diag(f'(net(t-1))))*(W*diag(f'(net(t-2))))*......*(W*diag(f'(net(K))))
        deltaWList.clear();
        deltaUList.clear();
        deltaWList.add(sensitivity_array);//t t-1 t-2,... 1
        deltaUList.add(sensitivity_array);

        //计算
        double[] deltaW =  sensitivity_array;
        double[] deltaU =  sensitivity_array;
        for(int i=times-1;i>0;i--)
        {
            double[][] t1= calcDelta_K(W, i);
            double[][] t2=CalcOperator.T(deltaW);
            double[][] t3=null;//CalcOperator.dot(t2, t1);
            double[] t4=CalcOperator._T(t3);

            deltaW =t4;
            deltaU =null;//CalcOperator._T(CalcOperator.dot(CalcOperator.T(deltaU),calcDelta_K(U,i)));

            deltaWList.add(deltaW);
            deltaUList.add(deltaU);
        }
    }

    /**
     * 根据k + 1 时刻的delta计算k时刻的delta
     * 公式：delta(K)=delta(T)*(W*diag(f'(net(t))))*(W*diag(f'(net(t-1))))*(W*diag(f'(net(t-2))))*......*(W*diag(f'(net(K))))
     * @param k
     */
    public double[][] calcDelta_K (double[][] W,int k)
    {
        double[] netK=netList.get(k);
        double[] f_netK=CalcOperator.relu_back(netK);
        double[][] diag_fNetK = CalcOperator.diag(f_netK);
        return CalcOperator.matrixMultiply(diag_fNetK,W);
    }

    /**
     * 计算公式: TiDu =Delta * Input
     *
     */
    public void calcGradient(double[] sensitivity_array)
    {
        for(int cnt=times;cnt>1;cnt--)
        {
            CalcOperator.fill(gradientU,0);

            CalcOperator.fill(gradientW, 0);

            for (int i = 1; i < cnt; i++)
            {
                for (int m = 0; m < gradientU.length; m++)
                {
                    for (int n = 0; n < gradientU[0].length; n++)
                    {
                        gradientU[m][n] += deltaUList.get(times-i)[m] *
                                inputList.get(i)[n];
                    }
                }
            }

            updateU();


            for (int i = 1; i < cnt; i++)
            {
                for (int m = 0; m < gradientW.length; m++)
                {
                    for (int n = 0; n < gradientW[0].length; n++)
                    {
                        gradientW[m][n] += deltaWList.get(times-i)[m] *
                                stateList.get(i - 1)[n];
                    }
                }
            }

            updateW();

            calcDelta(sensitivity_array);
        }
    }

    public void updateU()
    {
        // 按照梯度下降，更新权重
        for(int i=0;i<stateWidth;i++)
        {
            for(int j=0;j<inputWidth;j++)
            {
                if(learning_rate==0)
                {
                    U[i][j] -= (Math.pow(1/Math.E,i)) * gradientU[i][j];
                }
                else
                {
                    U[i][j] -= learning_rate * gradientU[i][j];
                }
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
                if(learning_rate==0)
                {
                    W[i][j] -= (Math.pow(1/Math.E,i)) * gradientW[i][j];
                }
                else
                {
                    W[i][j] -= learning_rate * gradientW[i][j];
                }
            }
        }
    }

    public void clear()
    {
        //当前时刻初始化为t0
        times=-1;
        stateList.clear();
        netList.clear();
        deltaUList.clear();
        deltaWList.clear();
        inputList.clear();
        stateList.add(lastState);
    }

    public void reset_state()
    {
        //当前时刻初始化为t0
        times=-1;
        stateList.clear();
        netList.clear();
        deltaUList.clear();
        deltaWList.clear();
        inputList.clear();
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
        RecurrentLayer recurrentLayer = new RecurrentLayer(2,2,0.2);

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

                double[] y3=recurrentLayer.getLastState();
                CalcOperator.print(y3);
                CalcOperator.print(t3);
                double err1 = error_function(CalcOperator.calcCECDelta(y3,t3));

                recurrentLayer.W[i][j]-=2*epsilon;
                recurrentLayer.reset_state();
                recurrentLayer.forward(x2);
                recurrentLayer.forward(x3);
                y3=recurrentLayer.getLastState();
                System.out.println("------------------------------------------------------------");
                CalcOperator.print(y3);
                CalcOperator.print(t3);
                double err2 = error_function(CalcOperator.calcCECDelta(y3,t3));

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
