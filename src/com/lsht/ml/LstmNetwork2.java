package com.lsht.ml;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import java.io.*;

/**
 * Created by Junson on 2017/8/24.
 */
public class LstmNetwork2
{
    public static String savePath = "LstmNetwork2Param.obj";

    //特征是 r,ra int inputWidth,int stateWidth
    //状态是 r,ra 的 one-hot 编码种类1694
    public LstmLayer2 lstmLayer2 = new LstmLayer2(14, 14, 0.01);

    public void train(double[][] data,int train_times)
    {
        for(int i=0;i<train_times;i++)
        {
            lstmLayer2.reset_state();

            for(int w=data.length-2;w>2;w--)
            {
                for (int k = 0; k <= w; k++)
                {
                    lastPredictResult = lstmLayer2.forward(data[k]);
                }

                lstmLayer2.backward(data[w+1]);
            }
        }

        File f = new File(savePath);

        try
        {
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(
                    new FileOutputStream(f));

            objectOutputStream.writeObject(lstmLayer2);
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

            lstmLayer2 = (LstmLayer2) objectInputStream.readObject();

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
        lastPredictResult = lstmLayer2.forward(lastPredictResult);

        return softToCode(lastPredictResult);
    }

    public String softToCode(double[] sr)
    {
        double max=sr[0];
        int indexMax=0;
        for(int i=1;i<7;i++)
        {
            if(max<sr[i])
            {
                max=sr[i];
                indexMax=i;
            }
        }

        StringBuilder sb=new StringBuilder("0000000");
        String ret=sb.replace(indexMax,indexMax+1,"1").toString();

        max=sr[7];
        indexMax=7;
        for(int i=8;i<14;i++)
        {
            if(max<sr[i])
            {
                max=sr[i];
                indexMax=i;
            }
        }

        StringBuilder sb2=new StringBuilder("0000000");
        String ret2=sb2.replace(indexMax-7,indexMax+1-7,"1").toString();

        return ret+ret2;
    }

    public String predict(double[] data)
    {
        lastPredictResult = lstmLayer2.forward(data);
        return softToCode(lastPredictResult);
    }

    public static void main(String[] args)
    {
        LstmNetwork2 rn = new LstmNetwork2();
        BiMap codeMap= HashBiMap.create();

        codeMap.put("b7",   "00000011000000");
        codeMap.put("我6",  "00000100100000");
        codeMap.put("准备5","00001000010000");
        codeMap.put("去4",  "00010000001000");
        codeMap.put("吃饭3","00100000000100");
        codeMap.put("了2",  "01000000000010");
        codeMap.put("了a",  "01000000000100");
        codeMap.put("吃饭b","00100000001000");
        codeMap.put("去c",  "00010000010000");
        codeMap.put("准备d","00001000100000");
        codeMap.put("我e",  "00000101000000");
        codeMap.put("e1",   "10000000000001");

        rn.lstmLayer2 = new LstmLayer2(14, 14,0.8);

        // rn.reLoad();

        String str = "b7 我6 准备5 去4 吃饭3 了2  e1 我6 准备5 去4 吃饭3 了2 我e 准备d 去c 吃饭b 了a  我e 准备d 去c 吃饭b 了a b7";// 去 吃饭 了  我 准备 去 吃饭 了  我 准备 去 吃饭 了 e
        rn.train(CalcOperator.strToCodeArray(str, codeMap), 1000);

        String  retCode=rn.predict(CalcOperator.strToCodeArray("准备d", codeMap)[0]);
        System.out.println(retCode + ":" + codeMap.inverse().get(retCode));
        String word="了a";
        for(int i=0;i<10;i++)
        {
            retCode=rn.predict(CalcOperator.strToCodeArray(word, codeMap)[0]);

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
