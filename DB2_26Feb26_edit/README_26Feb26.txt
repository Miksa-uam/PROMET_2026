This deliverable contains patient data from after March 2024, until up to 2025. 
In the extraction handed over between March to May 2025, the cutoff was February 2024. 
Since then, 1969 medical records from 1626 patients were collected, 
including about 75 patients treated with PNK CLINICS (PNK+GLP1). 

Some considerations:

1.
In the original 26Feb26 deliverable, previous data was not extracted, only these about 2000 patients. 
I manually merged the excels of the previous patients with these ones
to re-run the entire data standardization pipeline with this updated dataset. 
To make sure that the standardized SQLs from the previous batch are preserved,
I saved them under C:\Users\Felhasználó\Desktop\Projects\PNK_DB2\DB2_standard\data_upto_Feb24_delivered_23Apr25

2.
In this deliverable, there are no genomics, as as of now it is deactivated in the company. 

3. 
Some 6000 measurements only contain weight, no BMI or body composition. 
As such, they require special treatment.
As of 26Feb26, before saving the excels into SQL, I don't know how many patients are affected. 
However, if the measurements table is added to SQL as-is, it will generate missing data, 
and that might break any pipeline built on the assumption 
that the measurements table contains no missing BMI or body composition data. 