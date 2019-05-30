package com.lsht.ml;

/**
 * Created by Junson on 2017/8/21.
 */
public class ConvCalcOperator implements CalcOperator
{
    private ConvData convData;
    private FilterParam filterParam;

    public ConvCalcOperator(ConvData data,FilterParam filterParam)
    {
        this.convData=data;
        this.filterParam=filterParam;
        if(this.convData.getWeights()==null)
        {
            this.convData.setWeights(
                    new double[filterParam.getFilterCnt()][filterParam
                            .getFilterDeep()][filterParam
                            .getFilterHeight()][filterParam.getFilterWidth()]);
        }
    }

    private void initOutPutData()
    {
        convData.outputData=new double[filterParam.getFilterCnt()][][];

        int output_height = CalcOperator.calcPaddingSize(
                convData.inputData[0].length,
                convData.zeroPeddingSize, convData.stride,
                filterParam.getFilterHeight());
        int output_width = CalcOperator.calcPaddingSize(
                convData.inputData[0][0].length,
                convData.zeroPeddingSize, convData.stride,
                filterParam.getFilterWidth());
        for(int i=0;i<convData.outputData.length;i++)
        {
            convData.outputData[i]=new double[output_height][output_width];
        }
    }


    public void calcOutput()
    {
        initOutPutData();

        for(int o=0;o<convData.outputData.length;o++)
        {
            double[][][] weight=convData.weights[o];

            CalcOperator.conv(convData.zeroPeddingInputData,weight,convData.outputData[o],convData.stride,convData.bias[o]);
        }
    }

    public void initDelta()
    {
        int expanded_depth = convData.inputData.length;
        int expanded_height =  convData.inputData[0].length;
        int expanded_width = convData.inputData[0][0].length;

        convData.delta=new double[expanded_depth][expanded_height][expanded_width];
    }

    @Override
    public void calcDelta(double[][][] outputDelta)
    {
        initDelta();

        convData.outDelta=outputDelta;

        outputDelta=expand_sensitivity_map(outputDelta);

        bpSensitivityMap(outputDelta);
    }

    public double[][][] expand_sensitivity_map(double[][][] sensitivity_array)
    {
        int depth = convData.zeroPeddingInputData.length;

       // 确定扩展后sensitivity map的大小
       // 计算stride为1时sensitivity map的大小
       int expanded_width = (convData.zeroPeddingInputData[0][0].length -
                 convData.weights[0][0].length + 2 * convData.zeroPeddingSize + 1);
       int expanded_height = (convData.zeroPeddingInputData[0].length -
               convData.weights[0].length+ 2 * convData.zeroPeddingSize + 1);
       // # 构建新的sensitivity_map
       double expand_array[][][]  =new double[depth][expanded_height][expanded_width];

        // 从原始sensitivity map拷贝误差值
        for(int d=0;d<depth;d++)
        {
            for(int h=0;h<sensitivity_array[d].length;h++)
            {
                for(int w=0;w<sensitivity_array[d][h].length;w++)
                {
                    int i_pos = h * convData.stride;
                    int j_pos = w * convData.stride;

                    expand_array[d][i_pos][j_pos] = sensitivity_array[d][h][w];
                }
            }
        }

        return expand_array;
    }

    //处理卷积步长，对原始sensitivity map进行扩展
    public void calcGradient()
    {
        convData.weightsGrad=new double[convData.weights.length][convData.weights[0].length][convData.weights[0][0].length][convData.weights[0][0][0].length];
        //# 处理卷积步长，对原始sensitivity map进行扩展
        for(int i=0;i<convData.weights.length;i++)
        {
            for(int d=0;d<convData.delta.length;d++)
            {
                CalcOperator.conv(new double[][][]{convData.zeroPeddingInputData[d]},new double[][][]{convData.outDelta[i]},
                        convData.weightsGrad[i][d], 1, 0);
            }

            //计算偏置项的梯度是所有误差的和
            convData.biasGrad[i] = CalcOperator.sum(convData.outDelta[i]);
        }
    }

    /**
     * 计算传递到上一层的sensitivity map
     *  sensitivity_array: 本层的sensitivity map
     *  activator: 上一层的激活函数 这里统一用 relu函数,(relu的导数和自身一致)
     * @param sensitivity_array
     */
    public void bpSensitivityMap(double[][][] sensitivity_array)
    {
        for(int d=0;d<convData.delta.length;d++)
        {
            double[][] output=new double[convData.delta[d].length][convData.delta[d][0].length];


            double[][][] newWeights=new double[convData.weights.length][][];

            for(int i=0;i<convData.weights.length;i++)
            {
                // 将filter权重翻转180度
                double[][] flipped_weights = CalcOperator.flip(convData.weights[i][d]);
                newWeights[i]=flipped_weights;
            }

            CalcOperator.conv(sensitivity_array, newWeights,
                    output, 1, 0);

            convData.delta[d] = output;
        }
    }

    @Override
    public void updateWeigths(double rate)
    {
        calcGradient();

        for(int i=0;i<filterParam.getFilterCnt();i++)
        {
            for(int d=0;d<filterParam.getFilterDeep();d++)
            {
                for(int h=0;h<filterParam.getFilterHeight();h++)
                {
                    for(int w=0;w<filterParam.getFilterWidth();w++)
                    {
                        convData.weights[i][d][h][w]-=rate * convData.weightsGrad[i][d][h][w];
                    }
                }
            }
        }

        for(int i= 0;i<filterParam.getFilterCnt();i++)
        {
            convData.bias[i] -= rate * convData.biasGrad[i];
        }
    }
}
