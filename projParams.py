import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from sklearn.metrics          import classification_report, confusion_matrix, f1_score, accuracy_score,  recall_score
from sklearn.utils.multiclass import type_of_target, unique_labels

##########################################################################################################################################					
## defaults values for fourSquare API
##########################################################################################################################################					
version       = '20180724'                  # 
radius        = 25000                       # Radius in meters from the given location address
limit         = 1000                        # Maximum number of venues to return by foursquare
food_category = '4d4b7105d754a06374d81259'  # 'Root' category for all food-related venues

##########################################################################################################################################
projectSecrets = {	'clientAPIKey'        : '',
					'googleAPIkey'        : '',
					'4SquareClientID'     : '',
					'4SquareclientSecret' : ''}
					
##########################################################################################################################################					
## https://en.wikipedia.org/wiki/List_of_cuisines
##########################################################################################################################################
## List of Asian cuisines                                                https://en.wikipedia.org/wiki/List_of_Asian_cuisines
##########################################################################################################################################
cuisineType  = {'4bf58dd8d48988d142941735': 'eastAsianCuisine',          ## 'Asian Restaurant'        
               '4bf58dd8d48988d145941735' : 'eastAsianCuisine',          ## 'Chinese Restaurant'         
               '52af3a7c3cf9994f4e043bed' : 'eastAsianCuisine',          ## 'Cantonese Restaurant'       
               '58daa1558bbb0b01f18ec1d3' : 'eastAsianCuisine',          ## 'Cha Chaan Teng'             
               '4bf58dd8d48988d1f5931735' : 'eastAsianCuisine',          ## 'Dim Sum Restaurant'         
               '52af3a9f3cf9994f4e043bef' : 'eastAsianCuisine',          ## 'Dongbei Restaurant'         
               '4bf58dd8d48988d108941735' : 'eastAsianCuisine',          ## 'Dumpling Restaurant'                 
               '52af3aaa3cf9994f4e043bf0' : 'eastAsianCuisine',          ## 'Fujian Restaurant'                   
               '52af3ac83cf9994f4e043bf3' : 'eastAsianCuisine',          ## 'Hakka Restaurant'           
               '52af3afc3cf9994f4e043bf8' : 'eastAsianCuisine',          ## 'Hunan Restaurant'                    
               '4bf58dd8d48988d111941735' : 'eastAsianCuisine',          ## 'Japanese Restaurant'                                         
               '4bf58dd8d48988d113941735' : 'eastAsianCuisine',          ## 'Korean Restaurant'          
               '4bf58dd8d48988d1d1941735' : 'eastAsianCuisine',          ## 'Noodle House'                                               
               '52af3b463cf9994f4e043bfe' : 'eastAsianCuisine',          ## 'Peking Duck Restaurant'     
               '55a59bace4b013909087cb24' : 'eastAsianCuisine',          ## 'Ramen Restaurant'           
               '55a59bace4b013909087cb15' : 'eastAsianCuisine',          ## 'Shabu-Shabu Restaurant'     
               '52af3b593cf9994f4e043c00' : 'eastAsianCuisine',          ## 'Shanghai Restaurant'        
               '55a59bace4b013909087cb27' : 'eastAsianCuisine',          ## 'Soba Restaurant'            
               '52af3b773cf9994f4e043c03' : 'eastAsianCuisine',          ## 'Szechuan Restaurant'        
               '4bf58dd8d48988d1d2941735' : 'eastAsianCuisine',          ## 'Sushi Restaurant'                                            
               '52af3b813cf9994f4e043c04' : 'eastAsianCuisine',          ## 'Taiwanese Restaurant'       
               '55a59bace4b013909087cb2a' : 'eastAsianCuisine',          ## 'Udon Restaurant'            
               '52af3b913cf9994f4e043c06' : 'eastAsianCuisine',          ## 'Xinjiang Restaurant'        
               '4eb1d5724b900d56c88a45fe' : 'eastAsianCuisine',          ## 'Mongolian Restaurant'       
               '52af39fb3cf9994f4e043be9' : 'eastAsianCuisine',          ## 'Tibetan Restaurant'         
########################################################################################################################################
                '503288ae91d4c4b30a586d67' : 'southAsianCuisine',        ## 'Afghan Restaurant'             
                '54135bf5e4b08f3d2429dfe3' : 'southAsianCuisine',        ## 'Dosa Place'                    
                '52e81612bcbc57f1066b79fb' : 'southAsianCuisine',        ## 'Himalayan Restaurant'                         
                '4bf58dd8d48988d10f941735' : 'southAsianCuisine',        ## 'Indian Restaurant'             
                '5283c7b4e4b094cb91ec88d7' : 'southAsianCuisine',        ## 'Kebab Restaurant'              
                '54135bf5e4b08f3d2429dfdd' : 'southAsianCuisine',        ## 'North Indian Restaurant'       
                '52e81612bcbc57f1066b79f8' : 'southAsianCuisine',        ## 'Pakistani Restaurant'          
                '5413605de4b0ae91d18581a9' : 'southAsianCuisine',        ## 'Sri Lankan Restaurant'         
                '54135bf5e4b08f3d2429dfde' : 'southAsianCuisine',        ## 'South Indian Restaurant'       
########################################################################################################################################
                '56aa371be4b08b9a8d573568' : 'southEastAsianCuisine',    ## 'Burmese Restaurant'                                       
                '52e81612bcbc57f1066b7a03' : 'southEastAsianCuisine',    ## 'Cambodian Restaurant'                                     
                '4eb1bd1c3b7b55596b4a748f' : 'southEastAsianCuisine',    ## 'Filipino Restaurant'                                         
                '4deefc054765f83613cdba6f' : 'southEastAsianCuisine',    ## 'Indonesian Restaurant'                                   
                '4bf58dd8d48988d156941735' : 'southEastAsianCuisine',    ## 'Malay Restaurant'                  
                '56aa371be4b08b9a8d57350e' : 'southEastAsianCuisine',    ## 'Satay Restaurant'                  
                '4bf58dd8d48988d149941735' : 'southEastAsianCuisine',    ## 'Thai Restaurant'                                          
                '4bf58dd8d48988d14a941735' : 'southEastAsianCuisine',    ## 'Vietnamese Restaurant'             
########################################################################################################################################
                '5293a7d53cf9994f4e043a45' : 'westAsianCuisine',         ## 'Caucasian Restaurant'            
                '5bae9231bedf3950379f89e1' : 'westAsianCuisine',         ## 'Egyptian Restaurant'             
                '5283c7b4e4b094cb91ec88d8' : 'westAsianCuisine',         ## 'Doner Restaurant'                
                '4bf58dd8d48988d10b941735' : 'westAsianCuisine',         ## 'Falafel Restaurant'              
                '5bae9231bedf3950379f89e7' : 'westAsianCuisine',         ## 'Iraqi Restaurant'                
                '56aa371be4b08b9a8d573529' : 'westAsianCuisine',         ## 'Israeli Restaurant'              
                '52e81612bcbc57f1066b79fd' : 'westAsianCuisine',         ## 'Jewish Restaurant'               
                '5744ccdfe4b0c0459246b4ca' : 'westAsianCuisine',         ## 'Kurdish Restaurant'              
                '52e81612bcbc57f1066b79fc' : 'westAsianCuisine',         ## 'Kosher Restaurant'               
                '58daa1558bbb0b01f18ec1cd' : 'westAsianCuisine',         ## 'Lebanese Restaurant'             
                '4bf58dd8d48988d115941735' : 'westAsianCuisine',         ## 'Middle Eastern Restaurant'       
                '52e81612bcbc57f1066b79f7' : 'westAsianCuisine',         ## 'Persian Restaurant'              
                '5bae9231bedf3950379f89e4' : 'westAsianCuisine',         ## 'Shawarma Place'                  
                '5bae9231bedf3950379f89da' : 'westAsianCuisine',         ## 'Syrian Restaurant'               
                '4f04af1f2fb6e1c99f3db0bb' : 'westAsianCuisine',         ## 'Turkish Restaurant'              
                '5bae9231bedf3950379f89ea' : 'westAsianCuisine',         ## 'Yemeni Restaurant'     
##########################################################################################################################################
##   List of African cuisines                           https://en.wikipedia.org/wiki/List_of_African_cuisines  
##########################################################################################################################################
                '4bf58dd8d48988d1c8941735' : 'africanCuisine',           ## 'African Restaurant'              
                '4bf58dd8d48988d10a941735' : 'africanCuisine',           ## 'Ethiopian Restaurant'            
                '4bf58dd8d48988d1c3941735' : 'africanCuisine',           ## 'Moroccan Restaurant'             
##########################################################################################################################################
## List of European cuisines                            https://en.wikipedia.org/wiki/List_of_European_cuisines        
##########################################################################################################################################
               '52e81612bcbc57f1066b7a01' : 'centralEuroCuisine',        ## 'Austrian Restaurant'
               '56aa371be4b08b9a8d5734f3' : 'centralEuroCuisine',        ## 'Bulgarian Restaurant'
               '52f2ae52bcbc57f1066b8b81' : 'centralEuroCuisine',        ## 'Czech Restaurant'
               '4bf58dd8d48988d10d941735' : 'centralEuroCuisine',        ## 'German Restaurant'
               '52e81612bcbc57f1066b79fa' : 'centralEuroCuisine',        ## 'Hungarian Restaurant'
               '52e81612bcbc57f1066b7a04' : 'centralEuroCuisine',        ## 'Polish Restaurant'   
               '52960bac3cf9994f4e043ac4' : 'centralEuroCuisine',        ## 'Romanian Restaurant' 
               '56aa371be4b08b9a8d57355a' : 'centralEuroCuisine',        ## 'Slovak Restaurant' 
########################################################################################################################################
                '4bf58dd8d48988d109941735': 'eastNorthernEuroCuisine',   ## 'Eastern European Restaurant'
                '5293a7563cf9994f4e043a44': 'eastNorthernEuroCuisine',   ## 'Russian Restaurant'
                '52e928d0bcbc57f1066b7e96': 'eastNorthernEuroCuisine',   ## 'Ukrainian Restaurant'
                '52e81612bcbc57f1066b7a05': 'eastNorthernEuroCuisine',   ## 'English Restaurant'
                '5744ccdde4b0c0459246b4a3': 'eastNorthernEuroCuisine',   ## 'Scottish Restaurant'
                '4bf58dd8d48988d1c6941735': 'eastNorthernEuroCuisine',   ## 'Scandinavian Restaurant'
#########################################################################################################################################
               '58daa1558bbb0b01f18ec1ee' : 'southernEuroCuisine',       ## 'Bosnian Restaurant'
               '4bf58dd8d48988d10e941735' : 'southernEuroCuisine',       ## 'Greek Restaurant'
               '4bf58dd8d48988d110941735' : 'southernEuroCuisine',       ## 'Italian Restaurant'
               '4bf58dd8d48988d1c0941735' : 'southernEuroCuisine',       ## 'Mediterranean Restaurant'
               '4def73e84765ae376e57713a' : 'southernEuroCuisine',       ## 'Portuguese Restaurant'
               '4bf58dd8d48988d150941735' : 'southernEuroCuisine',       ## 'Spanish Restaurant'
               '4bf58dd8d48988d14d941735' : 'southernEuroCuisine',       ## 'Paella Restaurant'
               '4bf58dd8d48988d1db931735' : 'southernEuroCuisine',       ## 'Tapas Restaurant' 
##########################################################################################################################################
                '52e81612bcbc57f1066b7a02': 'westernEuroCuisine',        ## 'Belgian Restaurant'              
                '5744ccdfe4b0c0459246b4d0': 'westernEuroCuisine',        ## 'Dutch Restaurant'                
                '52e81612bcbc57f1066b7a09': 'westernEuroCuisine',        ## 'Fondue Restaurant'               
                '4bf58dd8d48988d10c941735': 'westernEuroCuisine',        ## 'French Restaurant'               
                '52e81612bcbc57f1066b79f9': 'westernEuroCuisine',        ## 'Modern European Restaurant'
                '4bf58dd8d48988d158941735': 'westernEuroCuisine',        ## 'Swiss Restaurant'          
##########################################################################################################################################
## List of American cuisines                            https://en.wikipedia.org/wiki/List_of_cuisines_of_the_Americas
##########################################################################################################################################
                '4bf58dd8d48988d14e941735': 'northAmericanCuisine',      ## 'American Restaurant'
                '4bf58dd8d48988d17a941735': 'northAmericanCuisine',      ## 'Cajun / Creole Restaurant'
                '4bf58dd8d48988d157941735': 'northAmericanCuisine',      ## 'New American Restaurant'
##########################################################################################################################################
                '4bf58dd8d48988d152941735': 'southAmericanCuisine',      ## 'Arepa Restaurant'          
                '4bf58dd8d48988d107941735': 'southAmericanCuisine',      ## 'Argentinian Restaurant'    
                '4bf58dd8d48988d16b941735': 'southAmericanCuisine',      ## 'Brazilian Restaurant'      
                '4bf58dd8d48988d153941735': 'southAmericanCuisine',      ## 'Burrito Place'                                                    
                '58daa1558bbb0b01f18ec1f4': 'southAmericanCuisine',      ## 'Colombian Restaurant'                                             
                '52939a8c3cf9994f4e043a35': 'southAmericanCuisine',      ## 'Empanada Restaurant'       
                '4bf58dd8d48988d1be941735': 'southAmericanCuisine',      ## 'Latin American Restaurant' 
                '4bf58dd8d48988d1c1941735': 'southAmericanCuisine',      ## 'Mexican Restaurant'        
                '4eb1bfa43b7b52c0e1adc2e8': 'southAmericanCuisine',      ## 'Peruvian Restaurant'       
                '5745c7ac498e5d0483112fdb': 'southAmericanCuisine',      ## 'Salvadoran Restaurant'                   
                '4bf58dd8d48988d1cd941735': 'southAmericanCuisine',      ## 'South American Restaurant' 
                '4bf58dd8d48988d151941735': 'southAmericanCuisine',      ## 'Taco Place'                
                '56aa371ae4b08b9a8d5734ba': 'southAmericanCuisine',      ## 'Tex-Mex Restaurant'                                               
                '56aa371be4b08b9a8d573558': 'southAmericanCuisine',      ## 'Venezuelan Restaurant'           
#########################################################################################################################################
                '52e81612bcbc57f1066b79fe': 'caribbeanCuisine',          ## 'Hawaiian Restaurant'        
                '4bf58dd8d48988d144941735': 'caribbeanCuisine',          ## 'Caribbean Restaurant'       
                '4bf58dd8d48988d154941735': 'caribbeanCuisine'           ## 'Cuban Restaurant'                                
}					

##########################################################################################################################################
##########################################################################################################################################
varList = { 'total_population'    : 'B01001_001E', 
            'median_age'          : 'B01002_001E', 
            'male_25_29'          : 'B01001_011E',
            'male_30_34'          : 'B01001_012E',
            'male_35_39'          : 'B01001_013E',
            'male_40_44'          : 'B01001_014E',
            'male_45_49'          : 'B01001_015E',
            'male_50_54'          : 'B01001_016E',
            'male_55_59'          : 'B01001_017E',
            'male_60_61'          : 'B01001_018E',
            'male_62_64'          : 'B01001_019E',
            'male_65_66'          : 'B01001_020E',
            'male_67_69'          : 'B01001_021E',
            'male_70_74'          : 'B01001_022E',
            'male_75_79'          : 'B01001_023E',
            'male_80_84'          : 'B01001_024E',
            'male_85_plus'        : 'B01001_025E',
            'female_25_29'        : 'B01001_035E',
            'female_30_34'        : 'B01001_036E',
            'female_35_39'        : 'B01001_037E',
            'female_40_44'        : 'B01001_038E',
            'female_45_49'        : 'B01001_039E',
            'female_50_54'        : 'B01001_040E',
            'female_55_59'        : 'B01001_041E',
            'female_60_61'        : 'B01001_042E',
            'female_62_64'        : 'B01001_043E',
            'female_65_66'        : 'B01001_044E',
            'female_67_69'        : 'B01001_045E',
            'female_70_74'        : 'B01001_046E',
            'female_75_79'        : 'B01001_047E',
            'female_80_84'        : 'B01001_048E',
            'female_85_plus'      : 'B01001_049E',
            'leave_630_7'         : 'B08011_006E',
            'leave_7_730'         : 'B08011_007E',
            'leave_730_8'         : 'B08011_008E',
            'leave_8_830'         : 'B08011_009E',
            'leave_830_9'         : 'B08011_010E',
            'walk_to_work'        : 'B08006_015E',
            'total_households'    : 'B11001_002E',
            'high_school_diploma' : 'B15003_017E',
            'bachelors_degree'    : 'B15003_022E',
            'masters_degree'      : 'B15003_023E',
            'median_hh_income'    : 'B19013_001E',
            'income_hh_50_60'     : 'B19001_011E',
            'income_hh_60_75'     : 'B19001_012E',
            'income_hh_75_100'    : 'B19001_013E',
            'income_hh_100_125'   : 'B19001_014E',
            'income_hh_125_150'   : 'B19001_015E',
            'income_hh_150_200'   : 'B19001_016E',
            'income_hh_200_plus'  : 'B19001_017E',
            'male_workers'        : 'B23022_003E',
            'female_workers'      : 'B23022_027E',
            'renter_occupied'     : 'B25008_003E',
            'median_rent'         : 'B25031_001E',
            'median_home_value'   : 'B25077_001E'}

##########################################################################################################################################
##########################################################################################################################################
varListTypes = {'state'             :int, 
				'county'            :int, 
				'county subdivision':int, 
				'place'             :int,
				'B01001_001E'       :float, 
				'B01002_001E'       :float, 
				'B25077_001E'       :float,
				'B01001_011E'       :float, 
				'B01001_012E'       :float, 
				'B01001_013E'       :float, 
				'B01001_014E'       :float, 
				'B01001_015E'       :float, 
				'B01001_016E'       :float, 
				'B01001_017E'       :float, 
				'B01001_018E'       :float, 
				'B01001_019E'       :float, 
				'B01001_020E'       :float, 
				'B01001_021E'       :float, 
				'B01001_022E'       :float, 
				'B01001_023E'       :float, 
				'B01001_024E'       :float, 
				'B01001_025E'       :float, 
				'B01001_035E'       :float, 
				'B01001_036E'       :float, 
				'B01001_037E'       :float, 
				'B01001_038E'       :float, 
				'B01001_039E'       :float, 
				'B01001_040E'       :float, 
				'B01001_041E'       :float, 
				'B01001_042E'       :float, 
				'B01001_043E'       :float, 
				'B01001_044E'       :float, 
				'B01001_045E'       :float, 
				'B01001_046E'       :float, 
				'B01001_047E'       :float, 
				'B01001_048E'       :float, 
				'B01001_049E'       :float, 
				'B08011_006E'       :float, 
				'B08011_007E'       :float, 
				'B08011_008E'       :float, 
				'B08011_009E'       :float, 
				'B08011_010E'       :float, 
				'B08006_015E'       :float, 
				'B11001_002E'       :float, 
				'B15003_017E'       :float, 
				'B15003_022E'       :float, 
				'B15003_023E'       :float, 
				'B19013_001E'       :float, 
				'B19001_011E'       :float, 
				'B19001_012E'       :float, 
				'B19001_013E'       :float, 
				'B19001_014E'       :float, 
				'B19001_015E'       :float, 
				'B19001_016E'       :float, 
				'B19001_017E'       :float, 
				'B23022_003E'       :float, 
				'B23022_027E'       :float, 
				'B25008_003E'       :float,
				'B25031_001E'       :float}
								
stateCountySubdivCols = ['state', 
                         'county', 
                         'county subdivision',
                         'B01001_001E', 
                         'B01002_001E', 
                         'B01001_011E', 
                         'B01001_012E', 
                         'B01001_013E', 
                         'B01001_014E',
                         'B01001_015E', 
                         'B01001_016E', 
                         'B01001_017E', 
                         'B01001_018E', 
                         'B01001_019E', 
                         'B01001_020E', 
                         'B01001_021E', 
                         'B01001_022E', 
                         'B01001_023E', 
                         'B01001_024E',
                         'B01001_025E', 
                         'B01001_035E', 
                         'B01001_036E', 
                         'B01001_037E', 
                         'B01001_038E', 
                         'B01001_039E', 
                         'B01001_040E', 
                         'B01001_041E', 
                         'B01001_042E', 
                         'B01001_043E', 
                         'B01001_044E', 
                         'B01001_045E', 
                         'B01001_046E', 
                         'B01001_047E', 
                         'B01001_048E', 
                         'B01001_049E', 
                         'B08011_006E', 
                         'B08011_007E', 
                         'B08011_008E', 
                         'B08011_009E', 
                         'B08011_010E', 
                         'B08006_015E', 
                         'B11001_002E', 
                         'B15003_017E', 
                         'B15003_022E', 
                         'B15003_023E', 
                         'B19013_001E', 
                         'B19001_011E', 
                         'B19001_012E', 
                         'B19001_013E', 
                         'B19001_014E', 
                         'B19001_015E', 
                         'B19001_016E', 
                         'B19001_017E', 
                         'B23022_003E', 
                         'B23022_027E', 
                         'B25008_003E', 
                         'B25031_001E', 
                         'B25077_001E']							 


##############################################################################################
###
##############################################################################################               
def plot_confusion_matrix(y_true, y_pred, classes, ax , normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
        
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    print(classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    print("\n")

    #fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax

def classifaction_report_csv(report,algo):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    reportFileName =  outDir + "class_report-"+ algo +".csv"
    dataframe.to_csv(reportFileName, index = False)            