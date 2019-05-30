package com.lsht.ml;

import java.util.ArrayList;
import java.util.List;

/**
 * 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
 * Created by Junson on 2017/8/20.
 */
public class Node
{
    private int layerIndex;
    private int nodeIndex;

    private List<Connection> upStreamConnections;
    private List<Connection> downStreamConnections;

    private double output;
    private double delta;

    private double sigmoid(double x)
    {
        return 1.0/(1+Math.pow(Math.E,-x));
    }

    public Node(int layer_index,int node_index)
    {
        this.layerIndex=layer_index;
        this.nodeIndex=node_index;
        downStreamConnections = new ArrayList();
        upStreamConnections =  new ArrayList();
        output = 0;
        delta = 0;
    }

    public void addDownstreamConnection(Connection connection)
    {
        downStreamConnections.add(connection);
    }

    public void addUpStreamConnection(Connection connection)
    {
        upStreamConnections.add(connection);
    }

    public double calcOutput()
    {
        output=0;

        for(int i=0;i<upStreamConnections.size();i++)
        {
            Connection conn=upStreamConnections.get(i);
            output+=conn.getUpstreamNode().getOutput()*conn.getWeight();
        }

        output = sigmoid(output);

        return output;
    }

    /**
     *  节点属于隐藏层时，根据式4计算delta
     * @return
     */
    public double calcHiddenLayerDelta()
    {

        double downstream_delta = 0;

        for(Connection conn : downStreamConnections)
        {
            downstream_delta += conn.getDownstreamNode().delta * conn.getWeight();
        }

        delta = output * (1 - output) * downstream_delta;

        return delta;
    }

    /**
     * 节点属于输出层时，计算delta
     * @param label
     * @return
     */
    public double calcOutputLayerDelta(double label)
    {
        delta = output * (1 - output) * (label - output);
        return delta;
    }


    public String toString()
    {
        return "output:"+output+";delta:"+delta+";upStreamConnects:"+upStreamConnections+";downStreamConnectis:"+downStreamConnections;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public int getLayerIndex() {
        return layerIndex;
    }

    public void setLayerIndex(int layerIndex) {
        this.layerIndex = layerIndex;
    }

    public int getNodeIndex() {
        return nodeIndex;
    }

    public void setNodeIndex(int nodeIndex) {
        this.nodeIndex = nodeIndex;
    }

    public List<Connection> getUpStreamConnections() {
        return upStreamConnections;
    }

    public void setUpStreamConnections(List<Connection> upStreamConnections) {
        this.upStreamConnections = upStreamConnections;
    }

    public List<Connection> getDownStreamConnections() {
        return downStreamConnections;
    }

    public void setDownStreamConnections(List<Connection> downStreamConnections) {
        this.downStreamConnections = downStreamConnections;
    }
}
