TSP_ID                              = "ID"
TSP_SHORT_ID                        = "Abbreviated ID (not safe)"

TSP_F_HZ                            = "Frequency (Hz)"
TSP_SAMPLE_B                        = "Size per sample (bits)"
TSP_LENGTH_S                        = "Length (s)"
TSP_START_S                         = "Start time (s)"
TSP_END_S                           = "End time (s)"
TSP_TIME_FORMAT                     = "Format in which the time is represented"
TSP_PHASE_DEG                       = "Sampling phase (°)"
TSP_OFFSET_B                        = "Input signal offset (bits)"
TSP_POWER_W                         = "Sampling power (W)"
TSP_EPC_J                           = "Energy per conversion (J)"
TSP_STEP_HISTORY                    = "Step history"
TSP_LATENCY_HISTORY                 = "Latency history"
TSP_SCORE_AIDI                      = "AIDI"
TSP_SCORE_DR_BPS                    = "Datarate (bps)"
TSP_SCORE_MSE_DB                    = "Mean Square Error (dB)"

TSP_INPUT_SERIES                    = "Input series"
TSP_OPERATION                       = "Operation"

TSP_SCORE_F1                        = "F-1 Score of task"
TSP_SCORE_FAR                       = "False-alarm rate Score of task"
TSP_SCORE_TIME_DIFF_AVG_S           = "Average time difference (s)"
TSP_SCORE_TIME_DIFF_STD_S           = "Std. deviation of time difference (s)"
TSP_SCORE_STATIC_PWR_W              = "Static power consumption (W)"
TSP_SCORE_STATIC_PWR_CPU_W          = "Static power consumption of the CPU (W)"
TSP_SCORE_DYN_PWR_CPU_FACTOR_W_MHZ  = "Dynamic power factor of the CPU (W/MHz)"
TSP_SCORE_ENERGY_PER_OP_J           = "Energy consumed per operation (J) - module-dependant"
TSP_SCORE_ENERGY_SAMPLING_J         = "Energy consumed during sampling (J)"
TSP_SCORE_ENERGY_FEX_J              = "Energy consumed during feature extraction (J)"
TSP_SCORE_ENERGY_CLASS_J            = "Energy consumed during classification (J)"
TSP_SCORE_ENERGY_TOTAL_J            = "Energy consumed during whole task (J)"
TSP_SCORE_ACQ_DR_BPS                = "Data rate of the acquisition stage (bps)"
TSP_SCORE_SMPL_DR_BPS               = "Data rate of the sampling stage (bps)"
TSP_SCORE_MEMORY_USAGE_B            = "Total memory usage over the execution period (b)"
TSP_SCORE_ASM_OPS                   = "Accumulated number of assmebly-code isntructions"
TSP_SCORE_MIN_FREQUENCY_HZ          = "Minimum frequency at which the system needs to run (Hz)"
TSP_SCORE_ENERGY_J                  = "Total energy consumption of the task (J)"
TSP_SCORE_POWER_W                   = "Total power consumption during the task (W)"
TSP_SCORE_EHRV_S                    = "Error of heart rate variability (s)"
TSP_SCORE_AVG_VECTOR_ERROR          = "Average error of RQ and RS vectors, neglecting offsets and scaling"
TSP_SCORE_MISSED_QRS                = "Missed QRS complexes"


TSP_SELECTED_SCORES                 = "Scores selected for total score"
TSP_SCORE_TOTAL                     = "Total score (product of selected scores)"

TSP_METHOD                          = "Acquisition method"


TSP_GROUND_TRUTH                    = "Ground truth series"

TSP_DET_PEAK_Q_SLOPE_DT_S           = "Q-wave max slope time"
TSP_DET_PEAK_Q_SLOPE_DX_REL         = "Q-wave relative amplitude change in dt_s"
TSP_DET_PEAK_R_SLOPE_DX_REL         = "R-wave relative amplitude change in dt_s"

TSP_CLASS_QRS_R_AMPL_MIN            = "QRS class, R-peak minimum amplitude relative"
TSP_CLASS_QRS_OFFSET_TOLERANCE_S    = "QRS class, peak offset tolerance (s)"

TSP_DET_FR_R_PEAK_DT_S              = "FR R-peak detection dt"
TSP_DET_FR_R_PEAK_DXDT_TH           = "FR R-peak detection dxdt threshold"
TSP_DET_FR_R_PEAK_AVG_TH            = "FR R-peak detection Average threshold"
TSP_DET_FR_R_PEAK_AVG_SCOPE         = "FR R-peak detection Average scope"

TSP_DET_LC_R_PEAK_LENGTH_N          = "LC R-peak detection Xings count"
TSP_DET_LC_R_PEAK_DT_S              = "LC R-peak detection Xings interval (s)"
TSP_DET_PEAK_LC_VAR_TIME            = "LC peak detection time variation"
TSP_DET_PEAK_LC_VAR_LEN             = "LC peak detection length variation"

TSP_DET_R_PEAK_SCORING_TOLERANCE    = "R-peak detection scoring tolerance"
TSP_DET_R_PEAK_SCORING_STRAT        = "R-peak detection scoring strategy"
TSP_DET_R_PEAK_SCORE_DET_RATE       = "R-peak detection score detection rate"
TSP_DET_R_PEAK_SCORE_FA_RATE_INV    = "R-peak detection score false alarm rate (1-far)"

TSP_DISTR_AIDIS_LIST                = "List of evaluated AIDIs"
TSP_DISTR_AIDIS_MARGIN              = "Margin to discard AIDI"

TSP_ADC_EPC_J                       = "Energy per conversion (J)"

TSP_LC_LVLS                         = "LC levels"
TSP_LC_LVL_W_FRACT                  = "LC level width by fraction"
TSP_LC_LVL_W_B                      = "LC level width by bits"
TSP_LC_STRAT                        = "LC strategy"
TSP_LC_FREQ_LIM                     = "LC ADC datarate limit"
TSP_LC_ACQ_AMP_B                    = "LC Acquisition word size of Amplitude"
TSP_LC_ACQ_DIR_B                    = "LC Acquisition word size of Direction"
TSP_LC_ACQ_TIME_B                   = "LC Acquisition word size of Time"
TSP_TIMER_F_HZ                      = "LC Timer frequency (Hz)"
TSP_LC_ACQ_AMP_STRAT                = "LC Acquisition strategy amplitude"
TSP_LC_ACQ_DIR_STRAT                = "LC Acquisition strategy direction"
TSP_LC_ACQ_TIME_STRAT               = "LC Acquisition strategy time"
TSP_LC_ACQ_F_HZ                     = "LC Acquisition ~ frequency (Hz)"
TSP_LC_REC_TIME                     = "LC Reconstruction (time only)"
TSP_LC_REC_METHOD                   = "LC reconstruction method"

TSP_AMPL_RANGE                      = "Amplitude range"
TSP_NOISE_DROP_RATE_DBPDEC          = "Noise drop rate (dB/dec)"
TSP_NOISE_DC_COMP                   = "DC component (???)"

TSP_INFODATA                        = "Informative data"
TSP_ANN                             = "Annotations (for informative data)"




TIME_FORMAT_DIFF_TM_N   = 'Number of timer periods after previous sample'
TIME_FORMAT_DIFF_FS_N   = 'Number of sampling periods after previous sample'
TIME_FORMAT_DIFF_S      = 'Time (s) after previous sample'
TIME_FORMAT_ABS_S       = 'Absolute time (s) of this sample'

METHODS_FRADC       = "FR-ADC"
METHODS_LCADC       = "LC-ADC"
METHODS_LCSUB       = "LCsubs"