package com.lsht.ml;

/**
 * Created by Junson on 2017/8/21.
 */
public class MeanPoolingCalcOperator extends PoolCalcOperator
{
    public MeanPoolingCalcOperator(ConvData data, int height,int width)
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
                    double sum = 0;
                    int cnt=0;
                    for (int i = 0;  i < convData.inputData[d].length; i += convData.stride)
                    {
                        for (int j = 0;  j < convData.inputData[d][i].length; j += convData.stride)
                        {
                            sum+=convData.inputData[d][i][j];
                            cnt++;
                        }
                    }

                    convData.outputData[d][h][w]=sum/cnt;
                }
            }
        }
    }
}
