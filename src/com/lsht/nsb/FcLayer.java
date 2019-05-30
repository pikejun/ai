package com.lsht.nsb;

import com.lsht.log.LoggerUtil;
import com.lsht.ml.CalcOperator;

/**
 * Created by Junson on 2017/9/8.
 */
public class FcLayer
{
    public String fcNetName;
    public String fcLayerName;
    public String unitName;
    public  double[][] weights;
    public int stateWidth;
    public int inputWidth;
    public double[] currentState;
    public double[][] currentInput;
    public double learnRate=0.2;
    public  double[][] currentGradient;

    public double currentGradientSum;

    public FcLayer(FcNetWork fcNetWork, String fcLayerName,String unitName,int stateWidth,int inputWidth,double learnRate)
    {
        this.fcNetName=fcNetWork.netName;

        this.fcLayerName=fcLayerName;

        this.unitName = unitName;

        this.stateWidth=stateWidth;

        this.inputWidth=inputWidth;

        this.learnRate= learnRate;

        this.weights=new double[stateWidth][inputWidth+1];
        this.currentGradient=new double[stateWidth][inputWidth+1];

        CalcOperator.randomUniform(-0.0001, 0.0001, weights);
    }

    public boolean saveWeights()
    {
       return NsbTools.saveWeights(fcNetName,fcLayerName,unitName,weights);
    }

    public double[] forward(double[][] input)
    {
        currentInput=NsbTools.addBiasForInput(input);

         double[] net=CalcOperator.dot(weights,currentInput);

        currentState = CalcOperator.sigmoid(net);

          return currentState;
    }

    public double[] backward(double[] delta)
    {
        double[] delta2=new double[currentInput[0].length-1];

         for(int i=0;i<currentGradient.length;i++)
         {
             for(int j=0;j<currentGradient[i].length;j++)
             {
                 currentGradient[i][j] = delta[i]*CalcOperator.sigmoid_d(currentState[i])*currentInput[i][j];
             }
         }

        double newSum = CalcOperator.sum(currentGradient);

        LoggerUtil.debugTrace(fcLayerName,"newGradientSum - currentGradientSum=" + newSum+"-"+ currentGradientSum+"="+(newSum -currentGradientSum));

        currentGradientSum = newSum;

        for(int w=0;w<delta2.length;w++)
        {
            double sum=0;

            for (int i = 0; i < delta.length; i++)
            {
                sum +=delta[i]*CalcOperator.sigmoid_d(currentState[i])*weights[i][w];
            }

            delta2[w]=sum;
        }

        return delta2;
    }

    public void updateWeights()
    {
        for(int i=0;i<weights.length;i++)
        {
            for(int j=0;j<weights[i].length;j++)
            {
                weights[i][j]-=learnRate*currentGradient[i][j];
            }
        }
    }

    public void reloadWeights()
    {
        weights = NsbTools.getWeights(fcNetName,fcLayerName,unitName);
    }

    public void updateWeightsToDB()
    {
        NsbTools.updateWeights(fcNetName,fcLayerName,unitName,weights);
    }
}
