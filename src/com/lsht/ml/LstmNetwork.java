package com.lsht.ml;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import java.io.*;

/**
 * Created by Junson on 2017/8/24.
 */
public class LstmNetwork
{
    public static String savePath = "LstmNetworkParam.obj";

    //特征是 r,ra int inputWidth,int stateWidth
    //状态是 r,ra 的 one-hot 编码种类1694
    public LstmLayer lstmLayer = new LstmLayer(7, 7, 0.01);

    public void train(double[][] data,int train_times)
    {
        for(int i=0;i<train_times;i++)
        {
            lstmLayer.reset_state();

            for(int w=data.length-2;w>2;w--)
            {
                for (int k = 0; k <= w; k++)
                {
                    lastPredictResult = lstmLayer.forward(data[k]);
                }

                lstmLayer.backward(data[w+1]);
            }
        }

        File f = new File(savePath);

        try
        {
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(
                    new FileOutputStream(f));

            objectOutputStream.writeObject(lstmLayer);
            objectOutputStream.close();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public boolean reLoad()
    {
        try
        {
            ObjectInputStream objectInputStream = new ObjectInputStream(
                    new FileInputStream(savePath));

            lstmLayer = (LstmLayer) objectInputStream.readObject();

            objectInputStream.close();
        }
        catch (Exception e)
        {
            e.printStackTrace();

            return false;
        }

        return true;
    }

    double[] lastPredictResult;

    public String predict()
    {
        lastPredictResult = lstmLayer.forward(lastPredictResult);
        int i = CalcOperator.getMaxIndex(lastPredictResult);
        return CalcOperator.toOneHotCode(lastPredictResult,i);
    }

    public String predict(double[] data)
    {
        lastPredictResult = lstmLayer.forward(data);
        int i = CalcOperator.getMaxIndex(lastPredictResult);
        return CalcOperator.toOneHotCode(lastPredictResult,i);
    }

    public static void main(String[] args)
    {
        LstmNetwork rn = new LstmNetwork();
        BiMap codeMap= HashBiMap.create();

        codeMap.put("b", "0000001");
        codeMap.put("我", "0000010");
        codeMap.put("准备", "0000100");
        codeMap.put("去", "0001000");
        codeMap.put("吃饭", "0010000");
        codeMap.put("了", "0100000");
        codeMap.put("e", "1000000");

        rn.lstmLayer = new LstmLayer(7, 7,0.8);

        // rn.reLoad();

        String str = "准备 去 b  我 准备 去 吃饭 了 我 准备 去 吃饭 了  e b 去 了";// 去 吃饭 了  我 准备 去 吃饭 了  我 准备 去 吃饭 了 e
        rn.train(CalcOperator.strToCodeArray(str, codeMap), 1);
        rn.predict(CalcOperator.strToCodeArray("b", codeMap)[0]);
        String word="我";
        for(int i=0;i<100;i++)
        {
            String retCode=rn.predict(CalcOperator.strToCodeArray(word, codeMap)[0]);

            word= ""+codeMap.inverse().get(retCode);
            System.out.println(retCode + ":" + word);
        }
        /*
        retCode=rn.predict(CalcOperator.strToCodeArray("准备", codeMap)[0]);
        System.out.println(retCode + ":" + codeMap.inverse().get(retCode));

        retCode=rn.predict(CalcOperator.strToCodeArray("去", codeMap)[0]);
        System.out.println(retCode + ":" + codeMap.inverse().get(retCode));

        retCode=rn.predict(CalcOperator.strToCodeArray("吃饭", codeMap)[0]);
        System.out.println(retCode + ":" + codeMap.inverse().get(retCode)); */
    }


}
