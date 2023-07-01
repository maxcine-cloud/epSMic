import argparse
def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('-dataType',type=str,help='test or train',default='test')
    parser.add_argument('-dbName',type=str,help='EOSM or SomaMutDB or COSMIC',default='EOSM')
    parser.add_argument('-dataPath',type=str,help='data path',default='./data')
    parser.add_argument('-processingmodelPath',type=str,help='processing model path',default='./model/data_processing_model')
    parser.add_argument('-intermodelPath',type=str,help='intermediate model path',default='./model/intermediate_model_all')
    parser.add_argument('-itermodelPath',type=str,help='iterUltimate model path',default='./model/iterUltimate_model_all')
    parser.add_argument('-processedPath',type=str,help='processed data path',default='./out/data')
    parser.add_argument('-interdataPath',type=str,help='intermediate data path',default='./out/inte_data')
    parser.add_argument('-iterdataPath',type=str,help='iterUltimate data path',default='./out/ulti_data')
    args=parser.parse_args()
    return args