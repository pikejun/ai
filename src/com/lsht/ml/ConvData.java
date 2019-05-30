package com.lsht.ml;

import java.util.Random;

/**
 * Created by Junson on 2017/8/21.
 */
public class ConvData
{
   public  static Random random=new Random();

   public double [][][] inputData;//输入数据
   public double [][][] zeroPeddingInputData;//加零Padding 后的数据

   public double [][][] outputData;//这个深度和Filter的个数相等


   public double [][][][] weights;//对应所有Filter的权重.
   public double [][][] delta;//对应输入数据的误差
   public double [][][] outDelta;//对应输出数据的误差
   public double [][][][] weightsGrad;//权重梯度,

   public double[] bias;//对应所有Filter的偏执
   public double[] biasGrad;//对应所有Filter的偏执梯度


   public int zeroPeddingSize=1;
   public int stride=1;

   public ConvData()
   {

   }

   public ConvData(double[][][][] weights)
   {
      setWeights(weights);
   }

   private void initParam()
   {
      if(weights==null)
      {
         return;
      }

      for(int i=0;i<weights.length;i++)
      {
         for (int d = 0; d < weights[i].length; d++)
         {
            for (int h = 0; h < weights[i][d].length; h++)
            {
               for (int w = 0; w < weights[i][d][h].length; w++)
               {
                  weights[i][d][h][w] =
                          random.nextDouble() * Math.pow(-1, random.nextInt()) *
                                  0.0001;
               }
            }
         }
      }

      bias=new double[weights.length];
      biasGrad=new double[weights.length];
   }


   public double[][][] getInputData()
   {
      return inputData;
   }

   public void setInputData(double[][][] inputData)
   {
       this.inputData = inputData;

       zeroPeddingInputData = CalcOperator.paddingInputData(inputData,zeroPeddingSize);
   }


   public double[][][] getOutputData()
   {
      return outputData;
   }

   public void setOutputData(double[][][] outputData)
   {
      this.outputData = outputData;
   }

   public double[][][] getDelta()
   {
      return delta;
   }

   public void setDelta(double[][][] delta)
   {
      this.delta = delta;
   }

   public double[][][][] getWeights()
   {
      return weights;
   }

   public void setWeights(double[][][][] weights)
   {
      this.weights = weights;

      initParam();
   }

   public double[] getBias()
   {
      return bias;
   }

   public void setBias(double[] bias)
   {
      this.bias = bias;
   }

   public int getZeroPeddingSize()
   {
      return zeroPeddingSize;
   }

   public void setZeroPeddingSize(int zeroPeddingSize)
   {
      this.zeroPeddingSize = zeroPeddingSize;
      zeroPeddingInputData =CalcOperator.paddingInputData(inputData,
              zeroPeddingSize);
   }

   public int getStride()
   {
      return stride;
   }

   public void setStride(int stride)
   {
      this.stride = stride;
   }
}
