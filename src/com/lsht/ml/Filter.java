package com.lsht.ml;

import java.util.Random;

/**
 * Created by Junson on 2017/8/21.
 */
public class Filter
{
    public  static Random random=new Random();

    private int filter_width;

    private int filter_height;

    private int channel_number;

    private double[][][] weights_grad;

    private double[][][] weights;

    private double bias;
    private double bias_grad;



    public Filter(int filter_width,
            int filter_height, int channel_number)
    {
        this.filter_width = filter_width;

        this.filter_height = filter_height;
        this.channel_number = channel_number;

        weights = new double[channel_number][filter_height][filter_width];

        for(int d=0;d<weights.length;d++)
        {
            for(int h=0;h<weights[d].length;h++)
            {
                for(int w=0;w<weights[d][h].length;w++)
                {
                    weights[d][h][w]=random.nextDouble()*Math.pow(-1,random.nextInt())*0.0001;
                }
            }
        }

        weights_grad =new double[channel_number][filter_height][filter_width];
        bias_grad = 0;
        bias = 0;
    }

    public void  update(double learning_rate)
    {
        for(int d=0;d<weights.length;d++)
        {
            for(int h=0;h<weights[d].length;h++ )
            {
                for(int w=0;w<weights[d][h].length;w++)
                {
                    weights[d][h][w] -= learning_rate * weights_grad[d][h][w];
                }
            }
        }

        bias -= learning_rate * bias_grad;
    }



    public int getFilter_width()
    {
        return filter_width;
    }

    public void setFilter_width(int filter_width)
    {
        this.filter_width = filter_width;
    }

    public int getFilter_height()
    {
        return filter_height;
    }

    public void setFilter_height(int filter_height)
    {
        this.filter_height = filter_height;
    }

    public int getChannel_number()
    {
        return channel_number;
    }

    public void setChannel_number(int channel_number)
    {
        this.channel_number = channel_number;
    }
}
