package com.lsht.ml;

/**
 * Created by Junson on 2017/8/21.
 */
public abstract class PoolCalcOperator implements CalcOperator
{
    public ConvData convData;

    public int width;

    public int height;

    public PoolCalcOperator(ConvData data,int height,int width)
    {
       this.convData=data;
       this.width=width;
       this.height=height;
    }

    public void initOutPutData()
    {
        this.convData.outputData=new double[convData.inputData.length][][];

        int output_height = CalcOperator.calcPaddingSize(
                convData.inputData[0].length,
                0, convData.stride, height);
        int output_width = CalcOperator.calcPaddingSize(
                convData.inputData[0][0].length,
                0, convData.stride, width);
        for(int i=0;i<convData.outputData.length;i++)
        {
            convData.outputData[i]=new double[output_height][output_width];
        }
    }

    public void initDelta()
    {

    }

    public void calcDelta(double[][][] delta)
    {
        initDelta();
    }

    @Override
    public void updateWeigths(double rate)
    {
       //noop
    }

    public int getWidth()
    {
        return width;
    }

    public void setWidth(int width)
    {
        this.width = width;
    }

    public int getHeight()
    {
        return height;
    }

    public void setHeight(int height)
    {
        this.height = height;
    }

    public ConvData getConvData()
    {
        return convData;
    }

    public void setConvData(ConvData convData)
    {
        this.convData = convData;
    }
}
