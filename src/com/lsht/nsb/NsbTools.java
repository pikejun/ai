package com.lsht.nsb;

import com.lsht.service.EurUsdService;
import com.lsht.service.NsbWeightsService;
import com.lsht.util.SpringContextHolder;

/**
 * Created by Junson on 2017/9/8.
 */
public class NsbTools
{
    private static EurUsdService eurUsdService;
    private static NsbWeightsService nsbWeightsService;

    public static void init()
    {
        if(eurUsdService==null)
        {
            eurUsdService = (EurUsdService)SpringContextHolder.getBean("eurUsdService");
        }

        if(nsbWeightsService==null)
        {
            nsbWeightsService = (NsbWeightsService)SpringContextHolder.getBean("nsbWeightsService");
        }
    }

    static
    {
        init();
    }

    public static boolean saveWeights(String netName,String layerName,String unitName,double[][] weights)
    {
        return nsbWeightsService.saveWeights(netName,layerName,unitName,weights);
    }
    public static double[][] getWeights(String netName,String layerName,String unitName)
    {
        return nsbWeightsService.getWeights(netName,layerName,unitName);
    }
    public static  void updateWeights(String netName, String layerName,
            String unitName,double[][] weights)
    {
        nsbWeightsService.updateWeights(netName, layerName, unitName, weights);
    }

    public static double[][] addBiasForInput(double[][] input)
    {
        double[][] ret=new double[input.length][];

        for(int i=0;i<input.length;i++)
        {
            int cols=input[i].length;

            ret[i]=new double[cols+1];

            int j=0;
            for(;j<cols;j++)
            {
                ret[i][j]=input[i][j];
            }

            ret[i][cols]=1.0;
        }

        return ret;
    }
}
