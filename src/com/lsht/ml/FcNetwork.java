package com.lsht.ml;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Junson on 2017/8/20.
 */
public class FcNetwork
{
    private Layer[] layers;

    private List<Connection> connections=new ArrayList<>();

    private int[] layersDef;

    /**
    初始化一个全连接神经网络
    layersDef:描述神经网络每层节点数
    */
    public FcNetwork(int[] layersDef) {
        this.layersDef=layersDef;

        this.layers = new Layer[layersDef.length];

        for (int i = 0; i < layersDef.length; i++) {
            layers[i] = new Layer(i, layersDef[i]);
        }

        for (int i = 0; i < layers.length - 1; i++) {
            Layer upLayer = layers[i];
            Layer downLayer = layers[i + 1];
            for (int j = 0; j < upLayer.getNodes().size(); j++) {
                for (int k = 0; k < downLayer.getNodes().size(); k++) {
                    connections.add(new Connection(upLayer.getNode(j), downLayer.getNode(k)));
                }
            }
        }
    }

    /**
     * 训练神经网络
     labels: 数组，训练样本标签。每个元素是一个样本的标签。
     data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
     * @param labels
     * @param data_set
     * @param rate
     * @param iteration
     */
    public void train(double[][] labels,double [][] data_set,double rate,int iteration)
    {
        for(int i=0;i<iteration;i++)
        {
            for(int d=0;d<data_set.length;d++)
            {
                train_one_sample(labels[d], data_set[d], rate);
            }
        }
    }

    /**
     *  内部函数，用一个样本训练网络
     * @param label
     * @param sample
     * @param rate
     */
    private void train_one_sample(double[] label,double[] sample,double rate)
    {
        predict(sample);
        calcDelta(label);
        updateWeight(rate);
    }

    /**
     * 内部函数，计算每个节点的delta
     * @param label
     * @return
     */
    public void calcDelta(double[] label)
    {
        List<Node> output_nodes = layers[layers.length-1].getNodes();

        for(int i=0;i<label.length;i++)
        {
            output_nodes.get(i).calcOutputLayerDelta(label[i]);
        }

        for(int i=layers.length-2;i>=0;i--)
        {
            output_nodes = layers[i].getNodes();

            for(int j=0;j<output_nodes.size();j++)
            {
                output_nodes.get(i).calcHiddenLayerDelta();
            }
        }
    }


    /**
     *  内部函数，更新每个连接权重
     * @param rate
     */
    public void updateWeight(double rate)
    {
        for(int i=layers.length-2;i>=0;i--)
        {
            for(Node node : layers[i].getNodes())
            {
                for(Connection conn :node.getDownStreamConnections())
                {
                    conn.updateWeight(rate);
                }
            }
        }
    }

    /**
     * 内部函数，计算每个连接的梯度
     */
    public void calcGradient()
    {
        for(int i=layers.length-2;i>=0;i--)
        {
            for(Node node : layers[i].getNodes())
            {
                for(Connection conn :node.getDownStreamConnections())
                {
                    conn.calcGradient();
                }
            }
        }
    }

    /**
     *
     *   获得网络在一个样本下，每个连接上的梯度
     label: 样本标签
     sample: 样本输入
     * @param label
     * @param sample
     * @return
     */
    public void getGradient(double[] label,double[] sample)
    {
        predict(sample);
        calcDelta(label);
        calcGradient();
    }

    /**
     *   根据输入的样本预测输出值
     * @param sample 数组，样本的特征，也就是网络的输入向量
     */
    public double[] predict(double[] sample)
    {
        layers[0].setOutputs(sample);
        for(int i=1;i<layers.length;i++)
        {
            layers[i].calcOutput();
        }

        return layers[layers.length-1].getOutputs();
    }

    public String toString()
    {
        StringBuilder sb=new StringBuilder();

        sb.append("layers length:"+layers.length);
        sb.append("\n");

        for(int i=0;i<layersDef.length;i++)
        {
            sb.append(" the ").append(i).append(" layer's node size=");
            sb.append(layersDef[i]).append("\n");
            sb.append(layers[i]);
            sb.append("\n");
        }

        return sb.toString();
    }

    public Layer[] getLayers() {
        return layers;
    }

    public void setLayers(Layer[] layers) {
        this.layers = layers;
    }

    public List<Connection> getConnections() {
        return connections;
    }

    public void setConnections(List<Connection> connections) {
        this.connections = connections;
    }

    public int[] getLayersDef() {
        return layersDef;
    }

    public void setLayersDef(int[] layersDef) {
        this.layersDef = layersDef;
    }

    public double networkError(double[] predictValues,double[] sample_label)
    {
        double error=0;

        for(int i=0;i<sample_label.length;i++)
        {
            error+=Math.pow(sample_label[i]-predictValues[i],2);
        }

        return error*0.5;
    }

    public static void main(String[] args)
    {
        FcNetwork nt=new FcNetwork(new int[]{2,6,3});

        //1 剪刀，2石头，3布
        //0 平，1，胜，2负
        double[][] data={{1,1},{1,2},{1,3},
                {2,1},{2,2},{2,3},
                {3,1},{3,2},{3,3}};

        double[][] labels={{0.9,0.01,0.01},{0.01,0.01,0.9},{0.01,0.9,0.01},{0.01,0.9,0.01},{0.9,0.01,0.01},{0.01,0.01,0.9},{0.01,0.01,0.9},{0.01,0.9,0.01},{0.9,0.01,0.01}};

        nt.train(labels, data, 0.01, 50000);

        double ret[]=nt.predict(new double[]{1, 1});
        System.out.println("1, 1=" + ret[0]+","+ ret[1]+","+ ret[2]);

        ret=nt.predict(new double[]{1, 2});
        System.out.println("1, 2=" + ret[0]+","+ ret[1]+","+ ret[2]);

        ret=nt.predict(new double[]{1, 3});
        System.out.println("1, 3=" + ret[0]+","+ ret[1]+","+ ret[2]);

        ret=nt.predict(new double[]{2, 1});
        System.out.println("2,1=" + ret[0]+","+ ret[1]+","+ ret[2]);

        ret=nt.predict(new double[]{2, 2});
        System.out.println("2,2=" + ret[0]+","+ ret[1]+","+ ret[2]);

        ret=nt.predict(new double[]{2, 3});
        System.out.println("2,3=" + ret[0]+","+ ret[1]+","+ ret[2]);

        ret=nt.predict(new double[]{3, 1});
        System.out.println("3,1=" + ret[0]+","+ ret[1]+","+ ret[2]);

        ret=nt.predict(new double[]{3, 2});
        System.out.println("3,2=" + ret[0]+","+ ret[1]+","+ ret[2]);

        ret=nt.predict(new double[]{3, 3});
        System.out.println("3,3=" + ret[0]+","+ ret[1]+","+ ret[2]);

        nt.gradient_check(nt,new double[]{3,3},new double[]{0.9,0.01,0.01});
        /*
        System.out.println("1, 2=" + nt.predict(new double[]{1,2})[0]);
        System.out.println("1, 3=" + nt.predict(new double[]{1, 3})[0]);

        System.out.println("2, 1=" + nt.predict(new double[]{2, 1})[0]);
        System.out.println("2,2=" + nt.predict(new double[]{2,2})[0]);
        System.out.println("2, 3=" + nt.predict(new double[]{2,3})[0]);

        System.out.println("3, 1=" + nt.predict(new double[]{3, 1})[0]);
        System.out.println("3, 2=" + nt.predict(new double[]{3,2})[0]);
        System.out.println("3, 3=" + nt.predict(new double[]{3,3})[0]); */

    }

    /**
     * 梯度检查
       network: 神经网络对象
       sample_feature: 样本的特征
       sample_label: 样本的标签
     * @param network
     * @param sample_feature
     * @param sample_label
     */
    public void  gradient_check(FcNetwork network,double[] sample_feature,double[] sample_label)
    {

        // 获取网络在当前样本下每个连接的梯度
        network.getGradient(sample_feature, sample_label);

        double epsilon = 0.0001;
        //对每个权重做梯度检查
        for(Connection conn: network.getConnections())
        {
            //获取指定连接的梯度
            double actual_gradient = conn.getGradient();
            //增加一个很小的值，计算网络的误差
            conn.setWeight(conn.getWeight()+epsilon);
            double error1 = networkError(network.predict(sample_feature), sample_label);

            // 减去一个很小的值，计算网络的误差
            // 刚才加过了一次，因此这里需要减去2倍
            conn.setWeight(conn.getWeight()-2*epsilon);
            double error2 = networkError(network.predict(sample_feature), sample_label);

            double expected_gradient = (error2 - error1) / (2 * epsilon);

            System.out.println(" expected gradient: \t"+expected_gradient+"\n actual gradient: \t"+actual_gradient);
        }

    }
}
