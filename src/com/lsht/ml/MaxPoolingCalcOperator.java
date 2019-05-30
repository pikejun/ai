package com.lsht.ml;

/**
 * Created by Junson on 2017/8/21.
 */
public class MaxPoolingCalcOperator extends PoolCalcOperator
{
    public MaxPoolingCalcOperator(ConvData data, int height,int width)
    {
        super(data, height,width);
    }

    @Override
    public void calcOutput()
    {
        initOutPutData();

        for (int d = 0; d < convData.outputData.length; d++)
        {
            for (int h = 0; h < convData.outputData[d].length; h++)
            {
                for (int w = 0; w < convData.outputData[d][h].length; w++)
                {
                    double max = 0;
                    for (int i = 0;  i < convData.inputData[d].length; i += convData.stride)
                    {
                        for (int j = 0;  j < convData.inputData[d][i].length; j += convData.stride)
                        {
                             if(max<convData.inputData[d][i][j])
                             {
                                 max=convData.inputData[d][i][j];
                             }
                        }
                    }

                    convData.outputData[d][h][w]=max;
                }
            }
        }
    }


}
