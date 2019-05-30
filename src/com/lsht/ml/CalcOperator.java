package com.lsht.ml;

import java.util.Map;
import java.util.Random;

/**
 * Created by Junson on 2017/8/21.
 */
public interface CalcOperator
{
    public void calcOutput();

    public void calcDelta(double[][][] delta);

    public void updateWeigths(double rate);

    public static int calcPaddingSize(int input_width,
            int zero_padding_cycle,
            int stride, int filter_width)
    {
        return (input_width + 2 * zero_padding_cycle - filter_width) / stride +
                1;
    }

    public static void conv(double[][][] input, double[][][] weight,
            double[][] output, int stride, double bias)
    {
        for (int i = 0; i < output.length; i++)
        {
            for (int j = 0; j < output[i].length; j++)
            {
                double sum = bias;

                for (int d = 0; d < input.length; d++)
                {
                    double k = 0;

                    for (int h = 0; h < weight[d].length; h++)
                    {
                        for (int w = 0; w < weight[d][h].length; w++)
                        {
                            k += input[d][h + i * stride][w + j * stride] *
                                    weight[d][h][w];
                        }
                    }

                    sum += k;
                }

                output[i][j] = CalcOperator.relu(sum);
            }
        }
    }

    public static double sum(double[] data)
    {
        double sum = 0;
        for (int i = 0; i < data.length; i++)
        {
            sum += data[i];
        }

        return sum;
    }

    public static double sum(double[][] data)
    {
        double sum = 0;
        for (int i = 0; i < data.length; i++)
        {
            for (int j = 0; j < data[i].length; j++)
            {
                sum += data[i][j];
            }
        }

        return sum;
    }

    public static double sum(double[][][] data)
    {
        double sum = 0;
        for (int i = 0; i < data.length; i++)
        {
            for (int j = 0; j < data[i].length; j++)
            {
                for (int k = 0; k < data[i][j].length; k++)
                {
                    sum += data[i][j][k];
                }
            }
        }

        return sum;
    }

    /**
     * 翻转180度输入
     *
     * @param input
     * @return
     */
    public static double[][][] flip(double[][][] input)
    {
        double[][][] ret = new double[input.length][input[0].length][input[0][0].length];

        for (int d = 0; d < input.length; d++)
        {
            for (int h = 0; h < input[0].length; h++)
            {
                for (int w = 0; w < input[0][0].length; w++)
                {
                    ret[d][h][w] = input[d][input[0].length - h - 1][
                            input[0][0].length - w - 1];
                }
            }
        }

        return ret;
    }

    public static double[][] flip(double[][] input)
    {
        double[][] ret = new double[input.length][input[0].length];

        for (int h = 0; h < input.length; h++)
        {
            for (int w = 0; w < input[0].length; w++)
            {
                ret[h][w] = input[input.length - h - 1][input[0].length - w -
                        1];
            }
        }

        return ret;
    }

    public static double[] add(double[] a, double[] b)
    {
        double[] ret = new double[a.length];

        for (int i = 0; i < a.length; i++)
        {
            ret[i] = a[i] + b[i];
        }

        return ret;
    }

    public static double[] minus(double[] a, double[] b)
    {
        double[] ret = new double[a.length];

        for (int i = 0; i < a.length; i++)
        {
            ret[i] = a[i] - b[i];
        }

        return ret;
    }

    public static double[] narrow(double[][] a)
    {
        double[] ret = new double[a.length];

        double sum = 0;
        for (int j = 0; j < a[0].length; j++)
        {
            for (int i = 0; i < a.length; i++)
            {
                sum += a[i][j];
            }

            ret[j] = sum;
        }

        return ret;
    }

    public static double[][] add(double[][] a, double[][] b)
    {
        double[][] ret = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++)
        {
            for (int j = 0; j < a[0].length; j++)
            {
                ret[i][j] = a[i][j] + b[i][j];
            }

        }

        return ret;
    }

    public static double[][][] add(double[][][] a, double[][][] b)
    {
        double[][][] ret = new double[a.length][a[0].length][a[0][0].length];

        for (int d = 0; d < a.length; d++)
        {
            for (int i = 0; i < a.length; i++)
            {
                for (int j = 0; j < a[0].length; j++)
                {
                    ret[d][i][j] = a[d][i][j] + b[d][i][j];
                }

            }
        }

        return ret;
    }

    public static double[][][] minus(double[][][] a, double[][][] b)
    {
        double[][][] ret = new double[a.length][a[0].length][a[0][0].length];

        for (int d = 0; d < a.length; d++)
        {
            for (int i = 0; i < a.length; i++)
            {
                for (int j = 0; j < a[0].length; j++)
                {
                    ret[d][i][j] = a[d][i][j] - b[d][i][j];
                }

            }
        }

        return ret;
    }

    public static double[][] minus(double[][] a, double[][] b)
    {
        double[][] ret = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++)
        {
            for (int j = 0; j < a[0].length; j++)
            {
                ret[i][j] = a[i][j] - b[i][j];
            }

        }

        return ret;
    }

    public static void print(double[] a)
    {
        for (int i = 0; i < a.length; i++)
        {
            if (i > 0)
            {
                System.out.print(",");
            }
            System.out.print(a[i]);
        }

        System.out.println();
    }

    public static void print(double[][] a)
    {
        for (int i = 0; i < a.length; i++)
        {
            if (i > 0)
            {
                System.out.println();
            }

            for (int j = 0; j < a[i].length; j++)
            {
                if (j > 0)
                {
                    System.out.print(",");
                }
                System.out.print(a[i][j]);
            }
        }

        System.out.println();
    }

    public static double[] toAarry(double a)
    {
        return new double[] { a };
    }

    public static double[] toAarry(double[] a)
    {
        double[] d = new double[a.length];
        System.arraycopy(a, 0, d, 0, a.length);

        return d;
    }

    public static double[] toAarry(double[][] a)
    {
        double[] d = new double[a.length * a[0].length];
        for (int i = 0; i < a.length; i++)
        {
            System.arraycopy(a[i], 0, d, i * a[0].length, a[0].length);
        }

        return d;
    }

    public static double[] toAarry(double[][][] a)
    {
        double[] ret = new double[a.length * a[0].length * a[0][0].length];
        for (int i = 0; i < a.length; i++)
        {
            for (int j = 0; j < a[0].length; j++)
            {
                System.arraycopy(a[i][j], 0, ret,
                        i * a[0].length * a[0][0].length + j * a[0][0].length,
                        a[0][0].length);
            }
        }

        return ret;
    }

    public static double[][][] paddingInputData(double[][][] inputData,
            int zeroPeddingSize)
    {
        if (inputData == null)
        {
            return null;
        }

        double[][][] zeroPeddingInputData = new double[inputData.length][][];

        int new_height = inputData[0].length + 2 * zeroPeddingSize;
        int new_width = inputData[0][0].length + 2 * zeroPeddingSize;

        for (int d = 0; d < zeroPeddingInputData.length; d++)
        {
            zeroPeddingInputData[d] = new double[new_height][new_width];

            for (int h = 0; h < inputData[d].length; h++)
            {
                for (int w = 0; w < inputData[d][h].length; w++)
                {
                    zeroPeddingInputData[d][h + zeroPeddingSize][w +
                            zeroPeddingSize] = inputData[d][h][w];
                }
            }
        }

        return zeroPeddingInputData;
    }

    /**
     * @param d
     * @return
     */
    public static double relu(double d)
    {
        if (d > 0)
        {
            return d;
        }
        else
        {
            return 0;
        }
    }

    public static double relu_d(double d)
    {
        if (d > 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    public static double[] relu_back(double[] d)
    {
        double[] ret = new double[d.length];
        for (int i = 0; i < d.length; i++)
        {
            if (d[i] > 0)
            {
                ret[i] = 1;
            }
            else
            {
                ret[i] = 0;
            }
        }

        return ret;
    }

    public static double[] relu(double[] d)
    {
        double[] ret = new double[d.length];
        for (int i = 0; i < d.length; i++)
        {
            ret[i] = relu(d[i]);
        }

        return ret;
    }

    public static void relu(double[][] d)
    {
        for (int i = 0; i < d.length; i++)
        {
            for (int j = 0; j < d[i].length; j++)
            {
                d[i][j] = relu(d[i][j]);
            }
        }
    }

    public static double[] dot(double[][] a, double[][] b)
    {
        double [] ret= new double[a.length];
        for(int i=0;i<a.length;i++)
        {
            double sum=0;
            for(int j=0;j<b[i].length;j++)
            {
                sum+=a[i][j]*b[i][j];
            }

            ret[i] =sum;
        }

        return ret;
    }

    /**
     * 目标函数。最小时最稳定
     *
     * @param y
     * @param label
     * @return
     */
    public static double targetE(double[] y, double[] label)
    {
        double ret = 0;

        for (int i = 0; i < y.length; i++)
        {
            ret += 0.5 * Math.pow((label[i] - y[2]), 2);
        }

        return ret;
    }

    public static int maxIndex(double[] y)
    {
        double ret = y[0];
        int index = 0;

        for (int i = 1; i < y.length; i++)
        {
            if (ret < y[i])
            {
                ret = y[i];
                index = i;
            }
        }

        return index;
    }

    /**
     * y 是one-hot编码
     *
     * @param y
     * @param label =[1,0,0,0]
     * @return
     */
    public static double targetL(double[] y, double[] label)
    {
        double ret = 0;

        int maxIndex = maxIndex(label);

        double max = softMax(y)[maxIndex];

        ret = -1 * Math.log(max);

        return ret;
    }

    /**
     * 当神经网络的输出层是softmax层时，对应的误差函数E通常选择交叉熵误差函数
     *
     * @param y
     * @return
     */
    public static double[] softMax(double[] y)
    {
        double[] ret = new double[y.length];

        double[] temp = new double[y.length];

        double sum = 0;
        for (int i = 0; i < y.length; i++)
        {
            temp[i] = Math.pow(Math.E, y[i]);
            sum += temp[i];
        }

        for (int i = 0; i < y.length; i++)
        {
            ret[i] = temp[i] / sum;
        }

        return ret;
    }

    /**
     * 交叉熵代价函数（cross-entropy cost function）
     * 当输出是softMax时，计算误差用交叉熵代价函数
     *
     * @return
     */
    public static double[] calcCECDelta(double[] y, double[] label)
    {

        double[] ret = new double[label.length];

        for(int i = 0; i < ret.length; i++)
        {
            ret[i] =  y[i] - label[i];
        }

        return ret;
    }

    public static double[] matrixMultiply(double a, double[] b)
    {
        double[] ret = new double[b.length];

        for (int i = 0; i < b.length; i++)
        {
            ret[i] = a * b[i];
        }
        return ret;
    }

    public static double[] matrixMultiply(double[] b, double a)
    {
        double[] ret = new double[b.length];

        for (int i = 0; i < b.length; i++)
        {
            ret[i] = a * b[i];
        }
        return ret;
    }

    /**
     * @param a (i*j)
     * @param b (j*k)
     * @return c(i*k)
     */
    public static double[] matrixMultiply(double[] a, double[] b)
    {
        double[] ret = new double[a.length];

        for (int i = 0; i < a.length; i++)
        {
            ret[i] = a[i] * b[i];
        }
        return ret;
    }

    public static void fill(double[] a, double fillNumber)
    {
        for (int i = 0; i < a.length; i++)
        {
            a[i] = fillNumber;
        }
    }

    public static String toOneHotCode(double[] ret, int indexHot)
    {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < ret.length; i++)
        {
            if (i == indexHot)
            {
                sb.append("1");
            }
            else
            {
                sb.append("0");
            }
        }

        return sb.toString();
    }

    public static int getMaxIndex(double[] d)
    {
        double t = d[0];
        int maxI = 0;

        for (int i = 1; i < d.length; i++)
        {
            if (t < d[i])
            {
                t = d[i];
                maxI = i;
            }
        }

        return maxI;
    }

    public static void fill(double[][] a, double fillNumber)
    {
        for (int i = 0; i < a.length; i++)
        {
            for (int j = 0; j < a[i].length; j++)
            {
                a[i][j] = fillNumber;
            }
        }
    }

    /**
     * @param a m*k
     * @param b k
     * @return m
     */
    public static double[] matrixMultiply(double[][] a, double[] b)
    {
        double[] ret = new double[a.length];

        for (int i = 0; i < a.length; i++)
        {
            double sum = 0;
            for (int j = 0; j < b.length; j++)
            {
                sum += a[i][j] * b[j];
            }

            ret[i] = sum;
        }

        return ret;
    }

    public static double[] dot(double[] b, double[] a)
    {
        double[] ret = new double[a.length];

        for (int i = 0; i < a.length; i++)
        {
             ret[i]=a[i]*b[i];
        }

        return ret;
    }


    public static double[][] dot(double[] b, double[][] a)
    {
        double[][] ret = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++)
        {
            for (int j = 0; j < b.length; j++)
            {
                ret[j][i] = a[j][i] * b[j];
            }
        }

        return ret;
    }

    /**
     * @param a (i*j)
     * @param b (j*k)
     * @return c(i*k)
     */
    public static double[][] matrixMultiply(double[][] a, double[][] b)
    {
        double[][] c = new double[a.length][b[0].length];

        for (int i = 0; i < a.length; i++)
        {
            for (int k = 0; k < b[0].length; k++)
            {
                for (int j = 0; j < a[0].length; j++)
                {
                    c[i][k] += a[i][j] * b[j][k];
                }
            }
        }

        return c;
    }

    public static double[] _T(double[][] a)
    {
        double[] ret = new double[a.length];
        for (int j = 0; j < a.length; j++)
        {
            ret[j] = a[j][0];
        }

        return ret;
    }

    public static double[][] T(double[] a)
    {
        double[][] ret = new double[a.length][1];
        for (int j = 0; j < a.length; j++)
        {
            ret[j][0] = a[j];
        }

        return ret;
    }

    public static double[][] T(double[][] a)
    {
        double[][] ret = new double[a[0].length][a.length];
        for (int i = 0; i < ret.length; i++)
        {
            for (int j = 0; j < ret[i].length; j++)
            {
                ret[i][j] = a[j][i];
            }
        }

        return ret;
    }

    public static double[][] diag(double[] a)
    {
        double[][] ret = new double[a.length][a.length];

        for (int i = 0; i < a.length; i++)
        {
            ret[i][i] = a[i];
        }

        return ret;
    }

    public static void main(String[] args)
    {
        double[][] a = { { 1, 2, 3, 4 }, { 4, 5, 6, 8 }, { 0, 1, 2, 3 },
                { 0, 1, 2, 3 } };
        double[][] b = { { 1, 2, 4, 0, 8 }, { 4, 5, 4, 0, 8 },
                { 3, 6, 4, 0, 8 }, { 3, 6, 4, 0, 8 } };

        print(matrixMultiply(a, b));
    }

    public static Random r = new Random(System.currentTimeMillis());

    public static double randomUniform(double from, double to)
    {
        double d2 = r.nextDouble();//(0,1)
        d2 = d2 * (to - from);//(0,to-from)
        return to - d2;
    }

    public static void randomUniform(double from, double to, double[] d)
    {
        for (int i = 0; i < d.length; i++)
        {
            d[i] = randomUniform(from, to);
        }
    }

    public static void randomUniform(double from, double to, double[][] d)
    {
        for (int i = 0; i < d.length; i++)
        {
            for (int j = 0; j < d[i].length; j++)
            {
                d[i][j] = randomUniform(from, to);
            }
        }
    }

    public static void randomUniform(double from, double to, double[][][] d)
    {

        for (int i = 0; i < d.length; i++)
        {
            for (int j = 0; j < d[i].length; j++)
            {
                for (int k = 0; k < d[i][j].length; k++)
                {
                    d[i][j][k] = randomUniform(from, to);
                }
            }
        }
    }

    public static double[] codeToCodeArray(String code)
    {
        double[] c = new double[code.length()];

        for (int j = 0; j < code.length(); j++)
        {
            try
            {
                c[j] = Double.valueOf(code.substring(j, j + 1));
            }
            catch (Exception e)
            {
                System.out.println(code + "       error:" + e.getMessage());
            }
        }

        return c;
    }

    public static double[][] strToCodeArray(String str, Map codeMap)
    {
        String[] strs = str.split("\\s+");

        double[][] ret = new double[strs.length][];

        for (int i = 0; i < strs.length; i++)
        {
            String code = String.valueOf(codeMap.get(strs[i]));
            ret[i] = codeToCodeArray(code);
        }

        return ret;
    }

    public static double sigmoid(double x)
    {
        return 1.0 / (1 + Math.pow(Math.E, -x));
    }

    public static double[] sigmoid(double x[])
    {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++)
        {
            ret[i] = 1.0 / (1 + Math.pow(Math.E, -x[i]));
        }
        return ret;
    }

    public static double sigmoid_d(double y)
    {
        return y * (1 - y);
    }

    public static double[] sigmoid_d(double y[])
    {
        double[] ret = new double[y.length];

        for (int i = 0; i < y.length; i++)
        {
            ret[i] = y[i] * (1 - y[i]);
        }
        return ret;
    }

    public static double tanh(double x)
    {
        return (Math.pow(Math.E, x) - Math.pow(Math.E, -x)) /
                (Math.pow(Math.E, x) + Math.pow(Math.E, -x));
    }

    public static double[] tanh(double[] x)
    {
        double[] ret = new double[x.length];

        for (int i = 0; i < x.length; i++)
        {
            ret[i] = (Math.pow(Math.E, x[i]) - Math.pow(Math.E, -x[i])) /
                    (Math.pow(Math.E, x[i]) + Math.pow(Math.E, -x[i]));
        }
        return ret;
    }

    public static double tanh_d(double y)
    {
        return 1 - y * y;
    }

    public static double[] tanh_d(double[] y)
    {
        double[] ret = new double[y.length];

        for (int i = 0; i < y.length; i++)
        {
            ret[i] = 1 - y[i] * y[i];
        }

        return ret;
    }

    /**
     * 1-a
     *
     * @return
     */
    public static double[] minusByOne(double[] a)
    {
        double[] ret = new double[a.length];

        for (int i = 0; i < a.length; i++)
        {
            ret[i] = 1 - a[i];
        }

        return ret;
    }

}
