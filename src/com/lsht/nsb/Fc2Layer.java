package com.lsht.nsb;

import com.lsht.log.LoggerUtil;
import com.lsht.ml.CalcOperator;

/**
 * Created by Junson on 2017/9/12.
 */
public class Fc2Layer
{
    public String netName;
    public String layerName;
    public String unitName;

    public double bias;
    public double currentGradient_b;

    public double w_power;
    public double currentGradient_p;
    public double w_r;
    public double currentGradient_r;
    public double w_t;
    public double currentGradient_t;
    public double currentState;
    public double learnRate=0.2;
    public double currentGradientSum;
    public double[] currentInput;

    public double a_net;

    public double w_a;
    public double currentGradient_a;
    public double w_c;
    public double currentGradient_c;

    public Fc2Layer(Fc2NetWork fc2NetWork, String fc2LayerName,String unitName,double learnRate)
    {
        this.netName=fc2NetWork.netName;

        this.layerName=fc2LayerName;

        this.unitName = unitName;

        this.learnRate= learnRate;

        double[] w=new double[3];

        CalcOperator.randomUniform(0.2, 0.4,w);

        w=CalcOperator.softMax(w);

        w_power=w[0];
        w_r=w[1];
        w_t=w[2];
        bias=CalcOperator.randomUniform(-0.00001, 0.00001);
        w_a=CalcOperator.randomUniform(0.95, 1.05);
        w_c=CalcOperator.randomUniform(-0.00001, 0.00001);
    }

    public boolean saveWeights()
    {
        double[][] weights=new double[1][6];
        weights[0][0]=w_power;
        weights[0][1]=w_r;
        weights[0][2]=w_t;
        weights[0][3]=bias;
        weights[0][4]=w_a;
        weights[0][5]=w_c;
        return NsbTools.saveWeights(netName,layerName,unitName,weights);
    }

    public double forward(double[] input)
    {
        currentInput=input;

        a_net=input[0]*w_power+input[1]*w_r+input[2]*w_t+bias;

        currentState = w_a*a_net+w_c;

        return currentState;
    }

    public double backward(double delta)
    {
        currentGradient_c=delta;
        currentGradient_a=delta*a_net;

        currentGradient_b=delta*w_a;
        currentGradient_p=currentGradient_b*currentInput[0];
        currentGradient_r=currentGradient_b*currentInput[1];
        currentGradient_t=currentGradient_b*currentInput[2];

        double newSum = currentGradient_c+currentGradient_a+currentGradient_p+currentGradient_r+currentGradient_t;

        LoggerUtil.debugTrace(layerName,
                "newGradientSum - currentGradientSum=" + newSum + "-" +
                        currentGradientSum + "=" +
                        (newSum - currentGradientSum));

        currentGradientSum = newSum;

        return delta;
    }

    public void updateWeights()
    {
        w_power-=learnRate*currentGradient_p;
        w_r-=learnRate*currentGradient_r;
        w_t-=learnRate*currentGradient_t;
        bias-=learnRate*currentGradient_b;
        w_a-=learnRate*currentGradient_a;
        w_c-=learnRate*currentGradient_c;

        if(Math.abs(w_power)>0.5)
        {
            w_power=0.33;
        }

        if(Math.abs(w_r)>0.5)
        {
            w_r=0.33;
        }

        if(Math.abs(w_t)>0.5)
        {
            w_t=0.33;
        }

        if(Math.abs(w_a)>2)
        {
            w_a=1;
        }

        if(Math.abs(w_c)>0.005)
        {
            w_c=0;
        }

        if(Math.abs(bias)>0.005)
        {
            bias=0;
        }
    }

    public void reloadWeights()
    {
       double[][] weights = NsbTools.getWeights(netName,layerName,unitName);

        w_power=weights[0][0];
        w_r=weights[0][1];
        w_t=weights[0][2];
        bias=weights[0][3];

        w_a=weights[0][4];
        w_c=weights[0][5];
    }

    public void updateWeightsToDB()
    {
        double[][] weights=new double[1][6];
        weights[0][0]=w_power;
        weights[0][1]=w_r;
        weights[0][2]=w_t;
        weights[0][3]=bias;
        weights[0][4]=w_a;
        weights[0][5]=w_c;
        NsbTools.updateWeights(netName,layerName,unitName,weights);
    }
}
