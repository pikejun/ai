package com.lsht.ml;

/**
 * Activator类实现了激活函数，其中，forward方法实现了前向计算，而backward方法则是计算导数
 * Created by Junson on 2017/8/21.
 */
public class ReluActivator
{
   public double forward(double weighted_input)
   {
       return Math.max(0,weighted_input);
   }

   public double  backward(double output)
   {
       if(output>0)
       {
           return 1;
       }
       else
       {
           return 0;
       }
   }

}
