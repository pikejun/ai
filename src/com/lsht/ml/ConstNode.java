package com.lsht.ml;


/**
 * 为了实现一个输出恒为1的节点(计算偏置项时需要)
 * Created by Junson on 2017/8/20.
 */
public class ConstNode extends Node
{
    public ConstNode(int layer_index, int node_index)
    {
        super(layer_index, node_index);
        setOutput(1);
    }

    public double calcHiddenLayerDelta()
    {
        return 0;
    }
}
