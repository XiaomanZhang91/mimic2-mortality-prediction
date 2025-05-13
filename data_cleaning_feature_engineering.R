library(tidyverse)
library(dplyr)
library(ggplot2)
library(tidyr)
library(purrr)
library(data.table)

############## 1. Data Exploration ###############
# MIMIC-II dataset stores time-series predictors for each subject in separate .txt files.
# Each file contains a "Parameter" and "Value" column over time.

### Reads and merges a few sample subjects to inspect variable availability

sub_1 <- fread("X/132539.txt")
sub_2 <- fread("X/132554.txt")
sub_3 <- fread("X/137162.txt")
sub_4 <- fread("X/137800.txt")

sub_1$SubjectID <- "132539"
sub_2$SubjectID <- "132554"
sub_3$SubjectID <- "137162"
sub_4$SubjectID <- "137800"

all_data <- rbind(sub_1, sub_2, sub_3, sub_4)

# Count how often each parameter appears for each subject
freq_table <- all_data %>%
  count(SubjectID, Parameter, name = "Frequency") %>%
  pivot_wider(
    names_from = Parameter,
    values_from = Frequency,
    values_fill = NA
  )

print(freq_table)

### Unique Variables from All Files
# This block scans all subject files to collect a complete list of measured variables.
# Since parameters vary across patients, this helps define the full feature space.

data_dir <- "X/"
files <- list.files(data_dir, pattern = "\\.txt$", full.names = TRUE)

# Read only the 'Parameter' column from each file to save memory
get_vars <- function(file) {
  dt <- fread(file, select = "Parameter", colClasses = "character")
  unique(dt$Parameter)
}

# Get unique variables across all files
all_vars <- map(files, get_vars)

# Flatten, remove duplicates, and exclude the ID variable
unique_vars <- sort(unique(unlist(all_vars)))
unique_vars <- setdiff(unique_vars, "RecordID")

# print(unique_vars)

# Clean up memory
remove(all_vars)


############## 2. Data cleaning & Feature Engineering ############
# Individual data were first stacked together
# Summaries statistics were calculated for each variables depending on their type (static or time-varying, continuous or categorical) and frequency(freqenct, moderate and rare).

##### 2.1 Stack individual data together into a long format #####

data_dir <- "X/"
files <- list.files(data_dir, pattern = "\\.txt$", full.names = TRUE)

static_vars <- c("Age","Gender","Height","ICUType","Weight")
freq_vars   <- c("HR","GCS","NISysABP","NIMAP","RespRate","Temp","Urine",
                 "DiasABP","FiO2","MAP","SysABP","NIDiasABP")
mod_vars    <- c("K","Mg","Na","WBC","PaCO2","PaO2","BUN","Creatinine",
                 "Glucose","HCO3","pH","HCT","Weight")
rare_vars   <- c("Platelets","ALP","ALT","AST","Albumin","Cholesterol",
                 "Lactate","TroponinI","TroponinT","Bilirubin","SaO2")
cat_vars    <- "MechVent"    

time_to_min <- function(tm) {
  parts <- strsplit(tm, ":", fixed=TRUE)[[1]]
  as.numeric(parts[1]) * 60 + as.numeric(parts[2])
}

long_dt <- rbindlist(map(files, ~ {
  dt <- fread(.x, colClasses="character")
  dt[, time_min := sapply(Time, time_to_min)]
  id <- dt[Parameter=="RecordID" & Time=="00:00", as.integer(Value)]
  dt[, `:=`(
    recordid = id,
    value = as.numeric(Value)
  )][
    Parameter != "RecordID",
    .(recordid, time_min, parameter = Parameter, value)
  ]
}))

# Tag time windows for the FREQUENT group
long_dt <- long_dt %>%
  filter(time_min <= 48*60) %>%
  mutate(window = case_when(
    time_min <= 6*60 ~ "0",
    time_min <= 24*60 ~ "6",
    TRUE ~ "24"
  ))

##### 2.2 Extract Baseline(static) vars #####
baseline <- long_dt %>%
  # pick only the very first chart time
  filter(time_min == 0, parameter %in% static_vars) %>%
  # in case there are duplicates at 00:00, take the first non‐NA
  group_by(recordid, parameter) %>%
  summarise(
    baseline = first(na.omit(value)),
    .groups = "drop"
  ) %>%
  pivot_wider(
    id_cols = recordid,
    names_from = parameter,
    values_from = baseline,
    names_glue = "{parameter}"
  )

##### 2.3 Feature Engineering for time-series measurements: continuous #####
# To handle the time series measurements, continuous variables were first classified into three groups based on their measurement frequency: frequent, moderate, and rare. For frequent variables (e.g., HR, GCS, blood pressure measurements), summary statistics including median, interquartile range (IQR), maximum, minimum, trend, number of observations (num_obs), first value, and missing indicators were calculated across three distinct time periods (0–6 hours, 6–24 hours, and 24–48 hours after ICU admission). For moderate variables (e.g., electrolytes, blood gases), the same set of statistics was generated over the entire 48-hour window. For rare variables (e.g., Platelets, AST, Troponin), a slightly reduced set of features—median, trend, num_obs, first value, and missing indicator—was extracted over the 48-hour period. 
# **Frequent Variables**  
# - Include: HR, GCS, NISysABP, NIMAP, RespRate, Temp, Urine, DiasABP, FiO₂, MAP, SysABP, NIDiasABP 
# - Summary statistics: median, IQR, maximum, minimum, trend, number of observations (`num_obs`), first value, and missing indicators  
# - Computed across three time periods: 0–6h, 6–24h, and 24–48h
full_grid <- expand.grid(
  recordid = unique(long_dt$recordid),
  parameter = freq_vars,
  window = c("0","6","24"),
  stringsAsFactors = FALSE
)

freq_sum <- long_dt %>%
  filter(parameter %in% freq_vars) %>%
  right_join(full_grid, by = c("recordid","parameter","window")) %>%
  group_by(recordid,parameter,window) %>%
  arrange(time_min) %>%
  summarise(
    median = median(value, na.rm = TRUE),
    IQR = IQR(value, na.rm = TRUE),
    min = min(value, na.rm = TRUE),
    max = max(value, na.rm = TRUE),
    first = first(na.omit(value)),
    trend = if(sum(!is.na(value))>=2) 
      last(na.omit(value))-first(na.omit(value)) 
    else NA_real_,
    n_obs = sum(!is.na(value)),
    missing = as.integer(n_obs==0)
  ) 

# **Moderate Variables:**  
# - Include: K, Mg, Na, WBC, PaCO₂, PaO₂, BUN, Creatinine, Glucose, HCO₃, pH, HCT, Weight
# - Summary statistics: median, IQR, maximum, minimum, trend, `num_obs`, first value, and missing indicator  
# - Computed over the **entire 48-hour period**
full_grid <- expand.grid(
  recordid = unique(long_dt$recordid),
  parameter = mod_vars,
  stringsAsFactors = FALSE
)
mod_sum <- long_dt %>%
  filter(parameter %in% mod_vars) %>%
  right_join(full_grid, by = c("recordid", "parameter")) %>%
  group_by(recordid, parameter) %>%
  arrange(time_min) %>%
  summarise(
    median = median(value, na.rm=TRUE),
    IQR = IQR(value, na.rm=TRUE),
    min = min(value, na.rm=TRUE),
    max = max(value, na.rm=TRUE),
    first = first(na.omit(value)),
    trend = if(sum(!is.na(value))>=2) 
      last(na.omit(value))-first(na.omit(value)) 
    else NA_real_,
    n_obs = sum(!is.na(value)),
    missing= as.integer(n_obs == 0)
  )

# **Rare Variables:**  
# - Include: Platelets, ALP, ALT, AST, Albumin, Cholesterol, Lactate, TroponinI, TroponinT, Bilirubin, SaO₂*  
# - Summary statistics: median, trend, `num_obs`, first value, and missing indicator  
# - Computed over the **entire 48-hour period**
full_grid <- expand.grid(
  recordid = unique(long_dt$recordid),
  parameter = rare_vars,
  stringsAsFactors = FALSE
)

# how many combos actually matched any dt row
long_dt %>% 
  filter(parameter %in% rare_vars) %>%
  distinct(recordid, parameter) %>%
  nrow()

rare_sum <- long_dt %>%
  filter(parameter %in% rare_vars) %>%
  right_join(full_grid, by = c("recordid", "parameter")) %>%
  group_by(recordid, parameter) %>%
  arrange(time_min) %>%
  summarise(
    median = median(value, na.rm = TRUE),
    first = first(na.omit(value)),
    trend = if(sum(!is.na(value))>=2) 
      last(na.omit(value))-first(na.omit(value)) 
    else NA_real_,
    n_obs = sum(!is.na(value)),
    missing = as.integer(n_obs == 0),
    .groups = "drop"
  )
remove(full_grid)
##### 2.4 Feature Engineering for time-series measurements: categorical #####
# MechVent (mechanical ventilation status) is the only time-varying categorical variable. 
# Since MechVent = 1 was the only recorded value (indicating the presence of mechanical ventilation)
# I summarized this variable at the patient level by computing:
# the total number of MechVent observations (n_obs), a binary indicator of whether the patient was ever ventilated (on) and identifying the first and last timestamps at which mechanical ventilation was recorded. From these, I further derived a categorical variable (duration_cat) reflecting the duration of mechanical ventilation, calculated as the time difference between the first and last MechVent records and grouped into clinically motivated intervals (0 = not ventilated; 1 = short; 2 = moderate; 3 = long; 4 = extended duration).
mech_sum <- long_dt %>%
  group_by(recordid) %>%
  summarise(
    n_obs = sum(parameter == "MechVent"),
    first = if (n_obs>0) min(time_min[parameter=="MechVent"]) else NA_real_,
    last = if (n_obs>0) max(time_min[parameter=="MechVent"]) else NA_real_
  ) %>%
  mutate(
    on = factor(ifelse(n_obs == 0, 0, 1), levels = c(0,1)),
    duration = last - first,
    duration = case_when(
      is.na(duration) ~ 0,
      duration <= 600 ~ 1,
      duration <= 1800 ~ 2,
      duration <= 2520 ~ 3,
      duration > 2520 ~ 4
    ),
    duration_cat = factor(duration, levels = 0:4)
  ) 

##### 2.5 Joint everything together #####
### Pivot each summary wide
pivot_feats <- function(df, stats, with_window = TRUE) {
  if (with_window) {
    df %>% pivot_wider(
      id_cols = recordid,
      names_from  = c("parameter","window"),
      values_from = all_of(stats),
      names_glue  = "{parameter}_{.value}_{window}"
    )
  } else {
    df %>% pivot_wider(
      id_cols = recordid,
      names_from = "parameter",
      values_from = all_of(stats),
      names_glue = "{parameter}_{.value}"
    )
  }
}

wide_freq <- pivot_feats(freq_sum, c("median","IQR","min","max","trend","first","n_obs","missing"), TRUE)
wide_mod  <- pivot_feats(mod_sum,  c("median","IQR","min","max","trend","first","n_obs","missing"), FALSE)
wide_rare <- pivot_feats(rare_sum, c("median","trend","first","n_obs","missing"),                 FALSE)
wide_mech <- mech_sum %>%
  rename_with(~ paste0(cat_vars, "_", .x), -recordid)

# Join all the X together
X_summary <- reduce(
  list(baseline, wide_freq, wide_mod, wide_rare, wide_mech),
  full_join, by="recordid"
)

# Join Y
Y <- fread("Y.txt")  
df_final <- X_summary %>%
  left_join(Y, by = c("recordid" = "RecordID"))

##### 2.6 Re-code for missing values #####

df_final <- df_final %>%
  mutate(across(ends_with("_n_obs"), ~ replace_na(.x, 0)),
         across(ends_with("_missing"), ~ replace_na(.x, 1)))
df_final <- df_final %>%
  mutate(across(ends_with(c("_n_obs_0", "_n_obs_6", "_n_obs_24")), ~ replace_na(.x, 0)),
         across(ends_with(c("_missing_0", "_missing_6", "_missing_24")), ~ replace_na(.x, 1)))

# Missing values are code as -1 for Gender, Height and Weight
# Check if there are -1 in other vars: no other vars has -1
# long_dt %>%
#   filter(value==-1) %>%
#   group_by(parameter) %>%
#   summarise(count = n())
df_final <- df_final %>%
  mutate(
    Gender = ifelse(Gender==-1, NA, Gender),
    Height = ifelse(Height == -1, NA, Height),
    Weight = ifelse(Weight == -1, NA, Weight)
  )

########### 3. Inspect & Double checked ################
##### 3.1 A quick overview #####
library(skimr)

sum_df <- skim(df_final)
sum_df %>% 
  as_tibble() %>% 
  arrange(desc(n_missing))

hist(sum_df$n_missing)

##### 3.2 Check an individual's data #####
subj_132539 <- fread("X/132539.txt")

library(stringr)
subj_132539 <- subj_132539 %>%
  mutate(
    hour = as.numeric(str_extract(Time, "^\\d{2}")) + as.numeric(str_extract(Time, "(?<=:)\\d{2}")) / 60
  )

subj_132539 <- subj_132539 %>%
  mutate(
    time_period = case_when(
      hour <= 6 ~ "0",
      hour <= 24 ~ "6",
      hour <= 48 ~ "24",
      TRUE ~ NA_character_
    )
  )
subj_132539 %>%
  filter(Parameter == "HR") %>%
  group_by(time_period) %>%
  summarise(
    HR_median = median(Value, na.rm = TRUE),
    HR_IQR = IQR(Value, na.rm = TRUE),
    HR_min = min(Value, na.rm = TRUE),
    HR_max = max(Value, na.rm = TRUE),
    HR_num_obs = sum(!is.na(Value)),
    HR_first = first(Value),
    HR_missing = sum(is.na(Value)),
    .groups = "drop"
  )

df_final %>% filter(recordid==132539) %>% select(starts_with("HR_"))
df_final %>% filter(recordid==137775) %>% select(Gender, Age)

##### 3.3 Check the overall summary statitsics #####
df_final2 <- df_final
df_final2 <- df_final2 %>%
  # remove vars for other part of the competition
  select(-c(`SAPS-I`, SOFA, Length_of_stay, Survival)) %>%
  mutate(
    Gender = as.factor(Gender),
    ICUType = as.factor(ICUType),
    across(matches("_missing(_[0-9]+)?$"), as.factor)
  )

# Median/percentage
static_median <- static_vars
freq_median <- unlist(lapply(freq_vars, function(x) {
  paste0(x, "_median_", c("0", "6", "24"))
}))
mod_median <- paste0(mod_vars, "_median")
rare_median <- paste0(rare_vars, "_median")

# Missing values
freq_missing <- unlist(lapply(freq_vars, function(x) {
  paste0(x, "_missing_", c("0", "6", "24"))
}))
mod_missing <- paste0(mod_vars, "_missing")
rare_missing <- paste0(rare_vars, "_missing")

# MechVent
mechvent_vars <- c( "MechVent_last", "MechVent_first", "MechVent_n_obs", "MechVent_on", "MechVent_duration", "MechVent_duration_cat")

vars_all <- c(
  static_median, 
  freq_median, 
  mod_median, 
  rare_median,
  freq_missing,
  mod_missing,
  rare_missing,
  mechvent_vars
)

df_final2 %>%
  select(all_of(vars_all)) %>%
  summary()
# remove(baseline,freq_sum,mech_sum, mod_sum,wide_freq,wide_mech, wide_mod,wide_rare,X_summary,files, get_vars, pivot_feats, time_to_min)

########### 3. Handling missing values ################
##### 3.1 Check for missing patterns #####
sum_df <- skim(df_final2)
sum_df %>% 
  as_tibble() %>% 
  arrange(desc(n_missing))

hist(sum_df$n_missing)

library(naniar)
df_sample <- df_final %>% slice_sample(n = 200)
# vis_miss(df_sample)

top_missing_vars <- df_final %>%
  summarise(across(everything(), ~mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_pct") %>%
  arrange(desc(missing_pct)) %>%
  slice_head(n = 20) %>%
  pull(variable) 

vis_miss(df_sample[, top_missing_vars])

##### 3.2 Handle missing in rare vars #####
# Most missing are coming from rare vars, because these tests were only ordered when a patient was in a special situation. 
# The strategy used here is to assume that patient who hadn't given such tests have normal test results. 
# Therefore, I impute the missing median and first value to be the midpoint of normal range of each measures, and missing trends were set to be 0. 
# Ranges of these measures are found by Gemini, with a few of them being doubled checked.
# Define normal ranges (lower, upper)
normal_ranges <- list(
  Platelets = c(150000, 400000),
  ALP = c(30, 120),
  ALT_male = c(10, 40),
  ALT_female = c(7, 35),
  AST_male = c(10, 34),
  AST_female = c(9, 32),
  Albumin = c(3.4, 5.4),
  Cholesterol = c(0, 200),
  Lactate = c(0, 2.0),
  TroponinI = c(0, 0.04),
  TroponinT = c(0, 0.015),
  Bilirubin = c(0.3, 1.0),
  SaO2 = c(95, 100)
)

midpoints <- sapply(normal_ranges, function(x) mean(range(x)))

# Midpoint helper
impute_mid <- function(x, lo, hi) ifelse(is.na(x), (lo + hi) / 2, x)

# Do all in one mutate
df_imputed <- df_final %>%
  mutate(
    #  STEP A: fill in all *_first 
    Platelets_first = impute_mid(Platelets_first, normal_ranges$Platelets[1], normal_ranges$Platelets[2]),
    ALP_first = impute_mid(ALP_first, normal_ranges$ALP[1], normal_ranges$ALP[2]),
    ALT_first = if_else(Gender==1,
                          impute_mid(ALT_first, normal_ranges$ALT_male[1], normal_ranges$ALT_male[2]),
                          impute_mid(ALT_first, normal_ranges$ALT_female[1], normal_ranges$ALT_female[2])),
    AST_first = if_else(Gender==1,
                              impute_mid(AST_first, normal_ranges$AST_male[1], normal_ranges$AST_male[2]),
                              impute_mid(AST_first, normal_ranges$AST_female[1], normal_ranges$AST_female[2])),
    Albumin_first = impute_mid(Albumin_first,normal_ranges$Albumin[1], normal_ranges$Albumin[2]),
    Cholesterol_first = impute_mid(Cholesterol_first, normal_ranges$Cholesterol[1], normal_ranges$Cholesterol[2]),
    Lactate_first = impute_mid(Lactate_first, normal_ranges$Lactate[1], normal_ranges$Lactate[2]),
    TroponinI_first = impute_mid(TroponinI_first, normal_ranges$TroponinI[1], normal_ranges$TroponinI[2]),
    TroponinT_first = impute_mid(TroponinT_first, normal_ranges$TroponinT[1], normal_ranges$TroponinT[2]),
    Bilirubin_first = impute_mid(Bilirubin_first, normal_ranges$Bilirubin[1], normal_ranges$Bilirubin[2]),
    SaO2_first = impute_mid(SaO2_first,normal_ranges$SaO2[1],      normal_ranges$SaO2[2])
  ) %>%
  mutate(
    # STEP B: wherever *_median is NA, copy the *_first
    across(ends_with("_median"), 
           ~ coalesce(.x, get(sub("_median", "_first", cur_column())))),
    # STEP C: wherever *_trend is NA, set to 0
    across(ends_with("_trend"), ~ coalesce(.x, 0))
  )

# Check missing
sum_df_imputed <- skim(df_imputed)
sum_df_imputed %>% arrange(desc(n_missing))
hist(sum_df_imputed$n_missing)

# Double check
# Rare var: ALP
#   recordid parameter n_obs missing
#      <int> <chr>     <int>   <int>
# 1   132539 ALP           0       1
df_imputed %>% 
  filter(recordid==132539) %>%
  select(starts_with("ALP"))

# Freq var: RespRate
df_imputed %>% 
  filter(recordid==132539) %>%
  select(starts_with("RespRate_"))
df_final %>% 
  filter(recordid==132539) %>%
  select(starts_with("RespRate_"))

df_imputed %>% 
  filter(recordid==132585) %>%
  select(starts_with("RespRate_"))
df_final %>% 
  filter(recordid==132585) %>%
  select(starts_with("RespRate_"))

##### 3.3 Handling subjects with most pred missing #####
# Check subjects with more most preds missing
subj_missing_num <-apply(df_imputed, 1, function(x) sum(is.na(x)))
hist(subj_missing_num)
dim(df_imputed) # 8000  461
max_missing_idx <- which(subj_missing_num == max(subj_missing_num)) #324 vars missing
# These 10 subjects only have stat var: 3148 3308 3438 4029 4129 4397 5159 5918 7008 7140
df_imputed[max_missing_idx, ] %>%
  select(-contains("missing"), -contains("n_obs")) %>%
  select(where(~ any(!is.na(.)))) 

# Exclude these 10 subjects from the dataset
df_imputed <- df_imputed[-c(3148, 3308, 3438, 4029, 4129, 4397, 5159, 5918, 7008, 7140),]
# Double check
# dim(df_imputed)
# df_imputed[3148:3149,]  