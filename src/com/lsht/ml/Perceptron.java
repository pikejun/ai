package com.lsht.ml;

import java.util.Random;
import java.util.function.Function;

/**
 * Created by Junson on 2017/8/20.
 */
public class Perceptron
{

    public static  Random random=new Random(System.currentTimeMillis());

    protected Function<Double,Double> activator;
    protected int inputNum;
    protected double[] weights;
    protected double bias;

    protected Perceptron()
    {

    }

    public Perceptron(int input_num)
    {
        init(input_num,this::sign);
    }

    public Perceptron(int input_num,Function<Double,Double> activator)
    {
        init(input_num,activator);
    }

    public void init(int input_num,Function<Double,Double> activator)
    {
        this.inputNum = input_num;

        this.activator=activator;

        this.weights =new double[input_num];

        for(int i=0;i<input_num;i++)
        {
            weights[i]=random.nextDouble() * Math.pow(-1, random.nextInt());
        }

        //偏置项初始化为0
        bias=0.0;
    }

    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        sb.append("weights :");
        for(int i=0;i<weights.length;i++)
        {
            if(i>0)
            {
                sb.append(",");
            }
            sb.append(weights[i]);
        }

        sb.append("bias ").append(bias);

        return sb.toString();
    }


    /**
     * 输入向量，输出感知器的计算结果
     * @param input_vec
     * @return
     */
    public double predict(double[] input_vec)
    {
        //  输入向量，输出感知器的计算结果
        // 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起

        double sum=bias;
        for(int i=0;i<input_vec.length;i++)
        {
            sum+=input_vec[i]*weights[i];
        }

        return activator.apply(sum);
    }

    /**
     * 输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
     * @param input_vecs
     * @param labels
     * @param iteration
     * @param rate
     */
    public void train(double[][] input_vecs,double [] labels,int iteration,double rate)
    {
        for(int i=0;i<iteration;i++)
        {
            //  一次迭代，把所有的训练数据过一遍
            _one_iteration(input_vecs, labels, rate);
        }

    }

    public void _one_iteration(double[][] input_vecs,double[] labels,double rate)
    {
        for(int i=0;i<input_vecs.length;i++)
        {
            // # 计算感知器在当前权重下的输出
            double input_vec[]=input_vecs[i];
            double output = predict(input_vec);
           // # 更新权重
           _update_weights(input_vec, output, labels[i], rate);
        }
    }

    public void _update_weights(double[] input_vec,double output,double label,double rate)
    {
        //按照感知器规则更新权重
        double  delta = label - output;
        for(int i=0;i<weights.length;i++)
        {
            weights[i]+=rate*delta*input_vec[i];
        }
        //更新bias
        bias += rate * delta;
    }

    public static void main(String[] args) {
        Perceptron p=new Perceptron(1);

        //y=3x+5;1.6666

        double[][] trainDataSet={{-1.6666},{1},{-5},{0},{0.4}};

        double[] label={0,1,-1,1,1};

        p.train(trainDataSet,label,1000,0.1);

        double[] test={-1.666699669};

        System.out.println(p.predict(test));
    }

    private double sign(double input)
    {
        if(input >0)
        {
            return 1;
        }
        else if(input==0)
        {
            return 0;
        }
        else
        {
            return -1;
        }
    }
}
