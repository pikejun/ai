package com.lsht.ml;

import java.util.Random;

/**
 * Connection对象，主要职责是记录连接的权重，以及这个连接所关联的上下游节点。
 * Created by Junson on 2017/8/20.
 */
public class Connection
{
    public static Random random=new Random(System.currentTimeMillis());
    private Node upstreamNode;
    private Node downstreamNode;
    private double weight;
    private double gradient;

    public Connection(Node upstream_node,Node downstream_node)
    {
        this.upstreamNode=upstream_node;
        upstream_node.addDownstreamConnection(this);
        this.downstreamNode=downstream_node;
        downstream_node.addUpStreamConnection(this);
        weight=random.nextDouble()*Math.pow(-1, random.nextInt());
        gradient = 0.0;
    }

    public void calcGradient()
    {
        gradient=downstreamNode.getDelta() * upstreamNode.getOutput();
    }


    public void updateWeight(double rate)
    {
        calcGradient();
        weight += rate * gradient;
    }

    public String toString()
    {
        StringBuilder sb=new StringBuilder();

        sb.append("upstreamNode:").append(upstreamNode).append(",downstreamNode:").append(downstreamNode);

        sb.append(",weight:").append(weight).append(",gradient:").append(gradient);

        return sb.toString();
    }


    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getGradient() {
        return gradient;
    }

    public void setGradient(double gradient) {
        this.gradient = gradient;
    }

    public Random getRandom() {
        return random;
    }

    public void setRandom(Random random) {
        this.random = random;
    }

    public Node getUpstreamNode() {
        return upstreamNode;
    }

    public void setUpstreamNode(Node upstreamNode) {
        this.upstreamNode = upstreamNode;
    }

    public Node getDownstreamNode() {
        return downstreamNode;
    }

    public void setDownstreamNode(Node downstreamNode) {
        this.downstreamNode = downstreamNode;
    }
}
