package com.lsht.ml;

import java.io.Serializable;

/**
 * Created by Junson on 2017/8/25.
 */
public class LstmLayer2 implements Serializable
{
    int inputWidth ;
    int stateWidth ;
    double learning_rate;

    double[][] W_fx;
    double[][] W_fh;
    double[] bias_f;
    double[] gradient_f;
    double[][] gradient_fx;
    double[][] gradient_fh;

    double[][] W_ix;
    double[][] W_ih;
    double[] bias_i;
    double[] gradient_i;
    double[][] gradient_ix;
    double[][] gradient_ih;


    double[][] W_cx;
    double[][] W_ch;
    double[] bias_c;
    double[] gradient_c;
    double[][] gradient_cx;
    double[][] gradient_ch;


    double[][] W_ox;
    double[][] W_oh;
    double[] bias_o;
    double[] gradient_o;
    double[][] gradient_ox;
    double[][] gradient_oh;

    int times=-1;

    double[] currentInput;
    double[] currentOutput;
    double[] currentState;
    double[] lastOutput;
    double[] lastState;

    public LstmLayer2(int input_width, int state_width, double learning_rate)
    {
        this.inputWidth=input_width;
        this.stateWidth=state_width;
        this.learning_rate=learning_rate;

        W_fx=new double[stateWidth][inputWidth];
        gradient_fx=new double[stateWidth][inputWidth];
        W_fh=new double[stateWidth][stateWidth];
        gradient_fh=new double[stateWidth][stateWidth];
        bias_f =new double[stateWidth];
        gradient_f=new double[stateWidth];

        W_ix=new double[stateWidth][inputWidth];
        gradient_ix=new double[stateWidth][inputWidth];
        W_ih=new double[stateWidth][stateWidth];
        gradient_ih=new double[stateWidth][stateWidth];
        bias_i =new double[stateWidth];
        gradient_i = new double[stateWidth];

        W_cx=new double[stateWidth][inputWidth];
        gradient_cx=new double[stateWidth][inputWidth];
        W_ch=new double[stateWidth][stateWidth];
        gradient_ch=new double[stateWidth][stateWidth];
        bias_c =new double[stateWidth];
        gradient_c = new double[stateWidth];

        W_ox=new double[stateWidth][inputWidth];
        gradient_ox=new double[stateWidth][inputWidth];
        W_oh=new double[stateWidth][stateWidth];
        gradient_oh=new double[stateWidth][stateWidth];
        bias_o =new double[stateWidth];
        gradient_o = new double[stateWidth];

        d_o=new double[stateWidth];
        d_c=new double[stateWidth];
        d_i=new double[stateWidth];
        d_f=new double[stateWidth];


        CalcOperator.randomUniform(-0.001, 0.001,W_fh);
        CalcOperator.randomUniform(-0.001, 0.001, W_fx);
        CalcOperator.randomUniform(-0.001, 0.001, bias_f);

        CalcOperator.randomUniform(-0.001, 0.001, W_ih);
        CalcOperator.randomUniform(-0.001, 0.001, W_ix);
        CalcOperator.randomUniform(-0.001, 0.001, bias_i);

        CalcOperator.randomUniform(-0.001, 0.001, W_ch);
        CalcOperator.randomUniform(-0.001, 0.001, W_cx);
        CalcOperator.randomUniform(-0.001, 0.001, bias_c);

        CalcOperator.randomUniform(-0.001, 0.001, W_oh);
        CalcOperator.randomUniform(-0.001, 0.001, W_ox);
        CalcOperator.randomUniform(-0.001, 0.001, bias_o);
    }

    double[] fo;
    double[] f_net;

    double[] fc_t_1_o;

    double[] io;
    double[] i_net;

    double[] c_net;
    double[] co;

    double[] ico;

    double[] o_net;
    double[] oo;

    double[] cto;

    double[] d_o;
    double[] d_c;
    double[] d_i;
    double[] d_f;


    public double[] forward(double[] input)
    {
        times++;
        currentInput = input;

        if(times==0)
        {
          // CalcOperator.sigmoid(CalcOperator.matrixMultiply(W_fx, input))
            i_net=CalcOperator
                    .add(bias_i, CalcOperator.matrixMultiply(W_ix, input));
            io= CalcOperator.sigmoid(i_net);

            c_net =CalcOperator.add(bias_c,
                    CalcOperator.matrixMultiply(W_cx, input));
            co = CalcOperator.tanh(c_net);
           currentState = CalcOperator.dot(io, co);

            o_net=CalcOperator.add(bias_o,
                    CalcOperator.matrixMultiply(W_ox, input));

            oo=CalcOperator.sigmoid(o_net);

            cto=CalcOperator.tanh(currentState);

            currentOutput =softMax2(CalcOperator.dot (cto, oo)) ;
        }
        else
        {
            f_net=CalcOperator
                    .add(bias_f, CalcOperator.matrixMultiply(W_fx, input));
            f_net=CalcOperator
                    .add(f_net, CalcOperator.matrixMultiply(W_fh, currentOutput));
            fo =CalcOperator.sigmoid(f_net);


            i_net=CalcOperator
                    .add(bias_i, CalcOperator.matrixMultiply(W_ix, input));
            i_net=CalcOperator
                    .add(i_net,
                            CalcOperator.matrixMultiply(W_ih, currentOutput));
            io =CalcOperator.sigmoid(i_net);


            c_net=CalcOperator
                    .add(bias_c, CalcOperator.matrixMultiply(W_cx, input));
            c_net=CalcOperator
                    .add(c_net,
                            CalcOperator.matrixMultiply(W_ch, currentOutput));
            co =CalcOperator.tanh(c_net);

            o_net=CalcOperator
                    .add(bias_o, CalcOperator.matrixMultiply(W_ox, input));
            o_net=CalcOperator
                    .add(o_net,
                            CalcOperator.matrixMultiply(W_oh, currentOutput));
            oo =CalcOperator.sigmoid(o_net);

            ico=CalcOperator.dot(io, co);

            fc_t_1_o=CalcOperator.dot(currentState, fo);

            lastState =currentState;

            currentState = CalcOperator.add(ico, fc_t_1_o);

            cto=CalcOperator.tanh(currentState);

            lastOutput = currentOutput;

            currentOutput = softMax2(CalcOperator.dot(cto, oo));
        }

        return currentOutput;
    }

    public static double[] softMax2(double[] y)
    {
        double[] a1=new double[7];
        double[] a2=new double[7];

        System.arraycopy(y,0,a1,0,7);
        System.arraycopy(y,7,a2,0,7);

        double[] s1=CalcOperator.softMax(a1);
        double[] s2=CalcOperator.softMax(a2);

        double[] ret=new double[14];

        System.arraycopy(s1,0,ret,0,7);
        System.arraycopy(s2,0,ret,7,7);

        return ret;
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

        updateWeight();//更新权重
    }



    /**
     * 计算公式: TiDu =Delta* Input
     *
     */
    public void calcGradient(double[] sensitivity_array)
    {
        CalcOperator.fill(gradient_fh,0);
        CalcOperator.fill(gradient_ih,0);
        CalcOperator.fill(gradient_ch,0);
        CalcOperator.fill(gradient_oh,0);

        CalcOperator.fill(gradient_fx,0);
        CalcOperator.fill(gradient_ix,0);
        CalcOperator.fill(gradient_cx,0);
        CalcOperator.fill(gradient_ox,0);

        CalcOperator.fill(gradient_f,0);
        CalcOperator.fill(gradient_i,0);
        CalcOperator.fill(gradient_c,0);
        CalcOperator.fill(gradient_o,0);

        for(int i=0;i<stateWidth;i++)
        {
            d_o[i]=sensitivity_array[i]*cto[i]*oo[i]*(1-oo[i]);
            d_c[i]=sensitivity_array[i]*oo[i]*(1-cto[i]*cto[i])*io[i]*(1-co[i]*co[i]);
            d_i[i]=sensitivity_array[i]*oo[i]*(1-cto[i]*cto[i])*co[i]*io[i]*(1-io[i]);
            d_f[i]=sensitivity_array[i]*oo[i]*(1-cto[i]*cto[i])*lastState[i]*fo[i]*(1-fo[i]);

            gradient_o[i] = d_o[i];
            gradient_c[i] = d_c[i];
            gradient_i[i] = d_i[i];
            gradient_f[i] = d_f[i];
        }

        for(int i=0;i<stateWidth;i++)
        {
            for(int j=0;j<stateWidth;j++)
            {
                double deltaOh=d_o[i];
                gradient_oh[i][j]=deltaOh*lastOutput[j];

                double deltaCh=d_c[i];
                gradient_ch[i][j]=deltaCh*lastOutput[j];

                double deltaIh=d_i[i];
                gradient_ih[i][j]=deltaIh*lastOutput[j];

                double deltaFh=d_f[i];
                gradient_fh[i][j]=deltaFh*lastOutput[j];
            }
        }

        for(int i=0;i<stateWidth;i++)
        {
            for(int j=0;j<inputWidth;j++)
            {
                double deltaOx=d_o[i];
                gradient_ox[i][j]=deltaOx*currentInput[j];

                double deltaCx=d_c[i];
                gradient_cx[i][j]=deltaCx*currentInput[j];

                double deltaIx=d_i[i];
                gradient_ix[i][j]=deltaIx*currentInput[i];

                double deltaFx=d_f[i];
                gradient_fx[i][j]=deltaFx*currentInput[j];
            }
        }
    }

    public void updateWeight()
    {
        for(int i= 0; i < stateWidth;i++)
        {
            for(int j=0;j<inputWidth;j++)
            {
                W_fx[i][j] -=learning_rate * gradient_fx[i][j];
                W_ix[i][j] -=learning_rate * gradient_ix[i][j];
                W_cx[i][j] -=learning_rate * gradient_cx[i][j];
                W_ox[i][j] -=learning_rate * gradient_ox[i][j];
            }
        }

        for(int i= 0; i < stateWidth;i++)
        {
            for(int j=0;j<stateWidth;j++)
            {
                W_fh[i][j] -=learning_rate * gradient_fh[i][j];
                W_ih[i][j] -=learning_rate * gradient_ih[i][j];
                W_ch[i][j] -=learning_rate * gradient_ch[i][j];
                W_oh[i][j] -=learning_rate * gradient_oh[i][j];
            }
        }

        bias_f =CalcOperator.minus(bias_f,CalcOperator.matrixMultiply(learning_rate,gradient_f));
        bias_i =CalcOperator.minus(bias_i,CalcOperator.matrixMultiply(learning_rate,gradient_i));
        bias_c =CalcOperator.minus(bias_c,CalcOperator.matrixMultiply(learning_rate,gradient_c));
        bias_o =CalcOperator.minus(bias_o,CalcOperator.matrixMultiply(learning_rate,gradient_o));
    }


    public void reset_state()
    {
        //当前时刻初始化为t0
        times=-1;
    }
}
