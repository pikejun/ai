package com.lsht.ml;

import java.util.ArrayList;
import java.util.List;

/**
 * Layer对象，负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作。
 * Created by Junson on 2017/8/20.
 */
public class Layer {

    protected int layerIndex;

    protected List<Node> nodes=new ArrayList<>();

    public Layer(int layer_index, int node_count)
    {
        this.layerIndex=layer_index;

        for(int i=0;i<node_count;i++)
        {
            nodes.add(new Node(layer_index, i));
        }

        nodes.add(new ConstNode(layer_index,node_count));
    }

    public void setOutPut(double[] data)
    {
        for(int i=0;i<nodes.size();i++)
        {
            nodes.get(i).setOutput(data[i]);
        }
    }

    public void calcOutput()
    {
        for(int i=0;i<nodes.size();i++)
        {
            nodes.get(i).calcOutput();
        }
    }

    public int getLayerIndex() {
        return layerIndex;
    }

    public void setLayerIndex(int layerIndex) {
        this.layerIndex = layerIndex;
    }

    public List<Node> getNodes() {
        return nodes;
    }

    public Node getNode(int i)
    {
        return nodes.get(i);
    }

    public void setNodes(List<Node> nodes) {
        this.nodes = nodes;
    }

    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        sb.append("layerIndex:").append(layerIndex).append(",nodeCount:").append(nodes.size());
        sb.append("\n").append(nodes);
        return sb.toString();
    }

    public void setOutputs(double[] data)
    {
        for(int i=0;i<data.length;i++)
        {
            nodes.get(i).setOutput(data[i]);
        }
    }

    public double[] getOutputs()
    {
        double[] ret=new double[nodes.size()];
        for(int i=0;i<nodes.size();i++)
        {
            ret[i]=nodes.get(i).getOutput();
        }

        return  ret;
    }
}
