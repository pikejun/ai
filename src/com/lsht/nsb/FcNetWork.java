package com.lsht.nsb;

import com.lsht.log.LoggerUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Junson on 2017/9/8.
 */
public class FcNetWork
{
    public  String netName="FcNetWork";

    public List<FcLayer> fcLayerList =new ArrayList<>();

    public double[] currentOutput;

    public double sumDelta;

    public FcNetWork(String netName)
    {
        this.netName = netName;
    }

    public double[] forward(double[][] input)
    {
        double[] s = fcLayerList.get(0).forward(input);

        for(int i=1;i<fcLayerList.size();i++)
        {

            FcLayer fcLayer =fcLayerList.get(i);

            int stateWidth=fcLayer.stateWidth;

            double[][] newInput=new double[stateWidth][];

            for(int w=0;w<stateWidth;w++)
            {
                newInput[w]=s;
            }

            s=fcLayer.forward(newInput);
        }

        currentOutput=s;

        return currentOutput;
    }

    public void backward(double[] labels)
    {
        double[] delta=new double[labels.length];

        for(int i=0;i<labels.length;i++)
        {
            delta[i]=currentOutput[i] -labels[i];
        }

        for(int i=fcLayerList.size()-1;i>=0;i--)
        {
            FcLayer fcLayer=fcLayerList.get(i);

            delta=fcLayer.backward(delta);
        }

        double newDeltaSum=0;
        for(int i=0;i<delta.length;i++)
        {
            newDeltaSum+=delta[i];
        }

        LoggerUtil.infoTrace(netName,"newDeltaSum - ="+newDeltaSum+"-"+sumDelta+"="+(newDeltaSum-sumDelta));

        sumDelta=newDeltaSum;
    }

    public void updateWeights()
    {
        for(int i=fcLayerList.size()-1;i>=0;i--)
        {
            FcLayer fcLayer=fcLayerList.get(i);

            fcLayer.updateWeights();
        }
    }


    public void reloadWeights()
    {
        for(int i=fcLayerList.size()-1;i>=0;i--)
        {
            FcLayer fcLayer=fcLayerList.get(i);

            fcLayer.reloadWeights();
        }
    }

    public void updateWeightsToDB()
    {
        for(int i=fcLayerList.size()-1;i>=0;i--)
        {
            FcLayer fcLayer=fcLayerList.get(i);

            fcLayer.updateWeightsToDB();
        }
    }


    public boolean saveWeights()
    {
        boolean ret=false;
        for(int i=fcLayerList.size()-1;i>=0;i--)
        {
            FcLayer fcLayer=fcLayerList.get(i);

            ret=fcLayer.saveWeights();

            if(!ret)
            {
                return ret;
            }
        }

        return ret;
    }
}
