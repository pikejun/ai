package com.lsht.nsb;

import com.lsht.log.LoggerUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Junson on 2017/9/12.
 */
public class Fc2NetWork
{
    public  String netName="Fc2NetWork";

    public List<Fc2Layer> fc2LayerList =new ArrayList<>();

    public double currentOutput;

    public double sumDelta;

    public Fc2NetWork(String netName)
    {
        this.netName = netName;
    }

    public double forward(double[] input)
    {
        double s = fc2LayerList.get(0).forward(input);

        for(int i=1;i<fc2LayerList.size();i++)
        {
            Fc2Layer fc2Layer =fc2LayerList.get(i);

            double t=fc2Layer.forward(input);

            if(Math.abs(t)>Math.abs(s))
            {
                s=t;
            }
        }

        currentOutput=s;

        return currentOutput;
    }

    public void backward(double labels)
    {
        double delta=currentOutput -labels;

        for(int i=fc2LayerList.size()-1;i>=0;i--)
        {
            Fc2Layer fc2Layer=fc2LayerList.get(i);

            delta=fc2Layer.backward(delta);
        }

        double newDeltaSum=0;
        newDeltaSum+=delta;

        LoggerUtil.infoTrace(netName,
                "newDeltaSum - =" + newDeltaSum + "-" + sumDelta + "=" +
                        (newDeltaSum - sumDelta));

        sumDelta=newDeltaSum;
    }

    public void updateWeights()
    {
        for(int i=fc2LayerList.size()-1;i>=0;i--)
        {
            Fc2Layer fc2Layer=fc2LayerList.get(i);

            fc2Layer.updateWeights();
        }
    }

    public void setLearnRate(double learnRate)
    {
        for(int i=fc2LayerList.size()-1;i>=0;i--)
        {
            Fc2Layer fc2Layer=fc2LayerList.get(i);

            fc2Layer.learnRate=learnRate;
        }
    }



    public void reloadWeights()
    {
        for(int i=fc2LayerList.size()-1;i>=0;i--)
        {
            Fc2Layer fc2Layer=fc2LayerList.get(i);

            fc2Layer.reloadWeights();
        }
    }

    public void updateWeightsToDB()
    {
        for(int i=fc2LayerList.size()-1;i>=0;i--)
        {
            Fc2Layer fc2Layer=fc2LayerList.get(i);

            fc2Layer.updateWeightsToDB();
        }
    }


    public boolean saveWeights()
    {
        boolean ret=false;
        for(int i=fc2LayerList.size()-1;i>=0;i--)
        {
            Fc2Layer fc2Layer=fc2LayerList.get(i);

            ret=fc2Layer.saveWeights();

            if(!ret)
            {
                return ret;
            }
        }

        return ret;
    }
}
