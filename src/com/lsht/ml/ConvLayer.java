package com.lsht.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Created by Junson on 2017/8/21.
 */
public class ConvLayer
{
    private double[][][] input_array;
    private int input_width;
    private int input_height;
    private int channel_number;

    private int filter_width;
    private int filter_height;


    private int zero_padding;
    private int stride;
    double learning_rate;



    private double[][][] padded_input_array;

    private ReluActivator activator;

    private List<Filter> filters;
    private int filter_number;
    private double[][][] output_array;
    private int output_width;
    private int output_height;

    public ConvLayer(int filter_number)
    {
        for (int i = 0; i < filter_number; i++)
        {
            this.filters.add(new Filter(filter_width,
                    filter_height, channel_number));
        }
    }

    public ConvLayer(int input_width, int input_height,
            int channel_number, int filter_width,
            int filter_height, int filter_number,
            int zero_padding, int stride, ReluActivator activator,
            double learning_rate)
    {
        this.input_width = input_width;
        this.input_height = input_height;
        this.channel_number = channel_number;
        this.filter_width = filter_width;
        this.filter_height = filter_height;
        this.filter_number = filter_number;
        this.zero_padding = zero_padding;
        this.stride = stride;
        this.output_width = calculate_output_size(
                input_width, filter_width, zero_padding,
                stride);
        this.output_height = calculate_output_size(
                input_height, filter_height, zero_padding,
                stride);

        this.output_array = new double[filter_number][output_height][output_width];
        filters = new ArrayList<>();
        for (int i = 0; i < filter_number; i++)
        {
            this.filters.add(new Filter(filter_width,
                    filter_height, channel_number));
        }

        this.activator = activator;
        this.learning_rate = learning_rate;
    }

    /**
     *  计算卷积层的输出
     输出结果保存在output_array
     */
    public void forward(double[][][] input_array)
    {
        this.input_array = input_array;
        this.padded_input_array = padding(input_array,
                 zero_padding);

        for(int i=0;i<filter_number;i++)
        {
           Filter f=filters.get(i);

          conv(padded_input_array,f,stride);

        }

        element_wise_op(output_array,activator::forward);
    }

    public void element_wise_op(double[][][] output_array,Function<Double,Double> func)
    {
        for(int d=0;d<output_array.length;d++)
        {
            for(int h=0;h<output_array[d].length;h++)
            {
                for(int w=0;w<output_array[d][h].length;w++)
                {
                    output_array[d][h][w]=func.apply(output_array[d][h][w]);
                }
            }
        }
    }

    public void conv(double[][][]padded_input_array,Filter f,int stride)
    {
        for(int d=0;d<padded_input_array.length;d++)
        {
            for(int h=0;h<padded_input_array[d].length;h+=stride)
            {
                for(int w=0;w<padded_input_array[d][h].length;w+=stride)
                {

                }
            }
        }



      /*  output_array[i][j] = (
                get_patch(input_array, i, j, kernel_width,
                        kernel_height, stride) * kernel_array
        ).sum() + bias */
    }

    public double[][][] padding(double[][][] input_array,int zero_padding)
    {
        double[][][] ret=new double[input_array.length][][];
        for(int d=0;d<input_array.length;d++)
        {
            ret[d]=new double[input_height+2*zero_padding][input_width+2*zero_padding];

            for(int h=0;h<input_array[d].length;h++)
            {
                for(int w=0;w<input_array[d][h].length;w++)
                {
                    ret[d][h+zero_padding][w+zero_padding]=input_array[d][h][w];
                }
            }
        }

        return ret;
    }


/*
    for f in range(self.filter_number):
    filter = self.filters[f]
    conv(self.padded_input_array,
            filter.get_weights(), self.output_array[f],
    self.stride, filter.get_bias())
    element_wise_op(self.output_array,
            self.activator.forward) */

    public int calculate_output_size(int input_width, int filter_width,
            int zero_padding,
            int stride)
    {
        return (input_width + 2 * zero_padding - filter_width) / stride + 1;
    }

    /**
     * @param d
     * @return
     */
    public double relu(double d)
    {
        if (d > 0)
        {
            return d;
        }
        else
        {
            return 0;
        }
    }

    public int getInput_width()
    {
        return input_width;
    }

    public void setInput_width(int input_width)
    {
        this.input_width = input_width;
    }

    public int getInput_height()
    {
        return input_height;
    }

    public void setInput_height(int input_height)
    {
        this.input_height = input_height;
    }

    public int getChannel_number()
    {
        return channel_number;
    }

    public void setChannel_number(int channel_number)
    {
        this.channel_number = channel_number;
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

    public int getFilter_number()
    {
        return filter_number;
    }

    public void setFilter_number(int filter_number)
    {
        this.filter_number = filter_number;
    }

    public int getZero_padding()
    {
        return zero_padding;
    }

    public void setZero_padding(int zero_padding)
    {
        this.zero_padding = zero_padding;
    }

    public int getStride()
    {
        return stride;
    }

    public void setStride(int stride)
    {
        this.stride = stride;
    }

    public double getLearning_rate()
    {
        return learning_rate;
    }

    public void setLearning_rate(double learning_rate)
    {
        this.learning_rate = learning_rate;
    }

    public int getOutput_width()
    {
        return output_width;
    }

    public void setOutput_width(int output_width)
    {
        this.output_width = output_width;
    }

    public int getOutput_height()
    {
        return output_height;
    }

    public void setOutput_height(int output_height)
    {
        this.output_height = output_height;
    }

    public double[][][] getOutput_array()
    {
        return output_array;
    }

    public void setOutput_array(double[][][] output_array)
    {
        this.output_array = output_array;
    }


    public List<Filter> getFilters()
    {
        return filters;
    }

    public void setFilters(List<Filter> filters)
    {
        this.filters = filters;
    }
}
