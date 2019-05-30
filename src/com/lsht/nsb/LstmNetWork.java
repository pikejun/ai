package com.lsht.nsb;

import com.lsht.log.LoggerUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Junson on 2017/9/8.
 */
public class LstmNetWork
{
    public  String netName="LstmNetWork";

    public List<LstmLayer> lstmLayerList = new ArrayList<>();

    public double[] currentOutput;

    public double[] sumDelta=new double[6];

    public  LstmNetWork(String name)
    {
        this.netName=name;
    }



    public double[] forward(double[] input)
    {  
        for(int i=0;i<lstmLayerList.size();i++)
        {

            LstmLayer lstmLayer =lstmLayerList.get(i);

            input=lstmLayer.forward(input);
        }

        currentOutput=input;

        return currentOutput;
    }

    public void backward(double[] labels)
    {
         /*
         * double[] sensitivity_array =CalcOperator.calcCECDelta(currentState,
         label);
         */

        double[] delta=new double[labels.length];

        for(int i=0;i<labels.length;i++)
        {
            delta[i]=currentOutput[i] -labels[i];
        }



        for(int i=lstmLayerList.size()-1;i>=0;i--)
        {
            LstmLayer lstmLayer=lstmLayerList.get(i);

            delta=lstmLayer.backward(delta);

            double newDeltaSum=0;
            for(int j=0;j<delta.length;j++)
            {
                if(!Double.isNaN(delta[j]))
                {
                    newDeltaSum+=delta[j];
                }
                else
                {
                    delta[j]=0;
                }
            }

            LoggerUtil.infoTrace(netName,
                    lstmLayer.layerName+"'s newDeltaSum - =" + newDeltaSum + "-" + sumDelta[i] + "=" +
                            (newDeltaSum - sumDelta[i]));

            sumDelta[i]=newDeltaSum;
        }
    }

    public void updateWeights()
    {
        for(int i=lstmLayerList.size()-1;i>=0;i--)
        {
            LstmLayer lstmLayer=lstmLayerList.get(i);

            lstmLayer.updateWeights();
        }
    }


    public void reloadWeights()
    {
        for(int i=lstmLayerList.size()-1;i>=0;i--)
        {
            LstmLayer lstmLayer=lstmLayerList.get(i);

            lstmLayer.reloadWeights();
        }
    }

    public void setLearnRate(double learnRate)
    {
        for(int i=lstmLayerList.size()-1;i>=0;i--)
        {
            LstmLayer lstmLayer=lstmLayerList.get(i);

            lstmLayer.setLearnRate(learnRate * (1+i*random.nextDouble()));
        }
    }

    public static Random random=new Random(System.currentTimeMillis());

    public void  reset_state()
    {
        for(int i=lstmLayerList.size()-1;i>=0;i--)
        {
            LstmLayer lstmLayer = lstmLayerList.get(i);

            lstmLayer.reset_state();
        }
    }

    public void updateWeightsToDB()
    {
        for(int i=lstmLayerList.size()-1;i>=0;i--)
        {
            LstmLayer lstmLayer=lstmLayerList.get(i);

            lstmLayer.updateWeightsToDB();
        }
    }


    public boolean saveWeights()
    {
        boolean ret=false;
        for(int i=lstmLayerList.size()-1;i>=0;i--)
        {
            LstmLayer lstmLayer=lstmLayerList.get(i);

            ret=lstmLayer.saveWeights();

            if(!ret)
            {
                return ret;
            }
        }

        return ret;
    }
}
