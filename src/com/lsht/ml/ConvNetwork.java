package com.lsht.ml;

import java.util.ArrayList;
import java.util.List;

/**
 * Input ->[ Conv*N+Pooling?]*M+FC*K
 * Created by Junson on 2017/8/21.
 */
public class ConvNetwork
{
   private List<FcNetwork> fcNetworks = new ArrayList<>();

   private List<ConvData> convDatas = new ArrayList<>();

   private List<CalcOperator> calcOperatorList=new ArrayList<>();

   public ConvNetwork()
   {

   }

    public void train(double[][][][] data_set,double[][][][] labels,double rate,int iteration)
    {
        for(int i=0;i<iteration;i++)
        {
            for(int d=0;d<data_set.length;d++)
            {
                train_one_sample(labels[d], data_set[d], rate);
            }
        }
    }


   public void train(double[][][][] data_set,double[][] labels,double rate,int iteration)
   {
       for(int i=0;i<iteration;i++)
       {
           for(int d=0;d<data_set.length;d++)
           {
               train_one_sample(labels[d], data_set[d], rate);
           }
       }
   }

    public void train_one_sample(double[][][] label,double[][][] data,double rate)
    {
        predict(data);

        calcDelta(label);

        updateWeigths(rate);
    }


   public void train_one_sample(double[] label,double[][][] data,double rate)
   {
       predict(data);

       calcDelta(label);

       updateWeigths(rate);
   }

    public void calcDelta(double[][][] label)
    {
        double[][][] t=convDatas.get(convDatas.size()-1).outputData;
        double[][][] wx=CalcOperator.minus(t,label);

        convDatas.get(convDatas.size()-1).outDelta=wx;
        calcOperatorList.get(convDatas.size()-1).calcDelta(wx);

        for(int i=convDatas.size()-2;i>=0;i--)
        {
            CalcOperator calcOperator=calcOperatorList.get(i);

            calcOperator.calcDelta(convDatas.get(i+1).getDelta());
        }
    }

    public void calcDelta(double[][] label)
    {
        double[][] t=convDatas.get(convDatas.size()-1).outputData[0];
        double[][] wx=CalcOperator.minus(t,label);

        convDatas.get(convDatas.size()-1).outDelta=new double[][][]{wx};

        for(int i=convDatas.size()-2;i>=0;i--)
        {
            CalcOperator calcOperator=calcOperatorList.get(i);

            calcOperator.calcDelta(convDatas.get(i+1).outDelta);
        }
    }

    public void calcDelta(double[] label)
    {
        double[] convLabel=calcFcDelta(label);

        double[][][] lastConvDataDelta=new double[convLabel.length][1][1];
        for(int d=0;d<convLabel.length;d++)
        {
            lastConvDataDelta[d][0][0] =convLabel[d];
        }

        convDatas.get(convDatas.size()-1).delta=lastConvDataDelta;

        for(int i=convDatas.size()-2;i>=0;i--)
        {
            CalcOperator calcOperator=calcOperatorList.get(i);

            calcOperator.calcDelta(convDatas.get(i+1).delta);
        }
    }


   public void  updateWeigths(double rate)
   {
       for(int i=fcNetworks.size()-1;i>=0;i--)
       {
           fcNetworks.get(i).updateWeight(rate);
       }

       for(int i=calcOperatorList.size()-1;i>=0;i--)
       {
           calcOperatorList.get(i).updateWeigths(rate);
       }
   }


   public double[] calcFcDelta(double[] label)
   {
       for(int i=fcNetworks.size()-1;i>=0;i--)
       {
           fcNetworks.get(i).calcDelta(label);

           label=getFcNetworksInputDelta(fcNetworks.get(i));
       }

       return label;
   }

   public double[] getFcNetworksInputDelta(FcNetwork fcNetwork)
   {
       Layer l=fcNetwork.getLayers()[0];
       double[] ret=new double[l.getNodes().size()];
       for(int i=0;i<l.getNodes().size();i++)
       {
           ret[i]=l.getNodes().get(i).getDelta();
       }

       return ret;
   }

   public double[] predict(double[][][] data)
   {
       convDatas.get(0).setInputData(data);

       for(int i=0;i<convDatas.size();i++)
       {
           if(i!=0)
           {
               convDatas.get(i).setInputData(convDatas.get(i-1).outputData);
           }

           calcOperatorList.get(i).calcOutput();
       }

       ConvData c=convDatas.get(convDatas.size()-1);

       double[] fcOutput=CalcOperator.toAarry(c.outputData);

       for(int i=0;i<fcNetworks.size();i++)
       {
           fcOutput = fcNetworks.get(i).predict(fcOutput);
       }

       return fcOutput;
   }

    public ConvNetwork add(CalcOperator calcOperator)
    {
        this.calcOperatorList.add(calcOperator);

        return this;
    }

    public ConvNetwork add(ConvData convData)
    {
        this.convDatas.add(convData);

        return this;
    }

    public ConvNetwork add(FcNetwork fcNetwork)
    {
        this.fcNetworks.add(fcNetwork);

        return this;
    }


    public static void main(String[] args)
    {
        ConvNetwork convNetwork=new ConvNetwork();
        convNetwork.add(new ConvData(new double[1][1][3][3]));
        convNetwork.add(new ConvData(new double[1][1][1][1]));
        convNetwork.add(new ConvCalcOperator(convNetwork.convDatas.get(0),new FilterParam(1,1,3,3)));
        convNetwork.add(new MaxPoolingCalcOperator(convNetwork.convDatas.get(1),2,2));

        convNetwork.train(new double[][][][] {

                        {
                                { { 10, 10, 10 } }
                        },
                        { { { 20, 20, 20 } } }
                },
                new double[][][][] {
                        { { { 1, 1, 1 } } },
                        { { { 2, 2, 2 } } } },
                0.01, 10000000);

        CalcOperator.print(
                convNetwork.predict(new double[][][] { { { 1, 1, 2 } } }));

        CalcOperator.print(
                convNetwork.predict(new double[][][] { { { 20,20, 20 } } }));

        CalcOperator.print(
                convNetwork.predict(new double[][][] { { { 20, 10, 20 } } }));
        CalcOperator.print(
                convNetwork.predict(new double[][][] { { { 10, 10, 10 } } }));
    }




}
