TS_PARAMS_ID                                = "ID"
TS_PARAMS_SHORT_ID                          = "Abbreviated ID (not safe)"

TS_PARAMS_F_HZ                              = "Frequency (Hz)"
TS_PARAMS_SAMPLE_B                          = "Size per sample (bits)"
TS_PARAMS_LENGTH_S                          = "Length (s)"
TS_PARAMS_START_S                           = "Start time (s)"
TS_PARAMS_END_S                             = "End time (s)"
TS_PARAMS_TIME_FORMAT                       = "Format in which the time is represented"
TS_PARAMS_PHASE_DEG                         = "Sampling phase (°)"
TS_PARAMS_OFFSET_B                          = "Input signal offset (bits)"
TS_PARAMS_POWER_W                           = "Sampling power (W)"
TS_PARAMS_EPC_J                             = "Energy per conversion (J)"
TS_PARAMS_STEP_HISTORY                      = "Step history"
TS_PARAMS_LATENCY_HISTORY                   = "Latency history"
TS_PARAMS_SCORE_AIDI                        = "AIDI"
TS_PARAMS_SCORE_DR_BPS                      = "Datarate (bps)"

TS_PARAMS_INPUT_SERIES                      = "Input series"
TS_PARAMS_OPERATION                         = "Operation"

TS_PARAMS_SCORE_F1                          = "F-1 Score of task"
TS_PARAMS_SCORE_FAR                         = "False-alarm rate Score of task"
TS_PARAMS_SCORE_TIME_DIFF_AVG_S             = "Average time difference (s)"
TS_PARAMS_SCORE_TIME_DIFF_STD_S             = "Std. deviation of time difference (s)"
TS_PARAMS_SCORE_STATIC_PWR_W                = "Static power consumption (W)"
TS_PARAMS_SCORE_STATIC_PWR_CPU_W            = "Static power consumption of the CPU (W)"
TS_PARAMS_SCORE_DYN_PWR_CPU_FACTOR_W_MHZ    = "Dynamic power factor of the CPU (W/MHz)"
TS_PARAMS_SCORE_ENERGY_PER_OP_J             = "Energy consumed per operation (J) - module-dependant"
TS_PARAMS_SCORE_ENERGY_SAMPLING_J           = "Energy consumed during sampling (J)"
TS_PARAMS_SCORE_ENERGY_FEX_J                = "Energy consumed during feature extraction (J)"
TS_PARAMS_SCORE_ENERGY_CLASS_J              = "Energy consumed during classification (J)"
TS_PARAMS_SCORE_ENERGY_TOTAL_J              = "Energy consumed during whole task (J)"
TS_PARAMS_SCORE_ACQ_DR_BPS                  = "Data rate of the acquisition stage (bps)"
TS_PARAMS_SCORE_SMPL_DR_BPS                 = "Data rate of the sampling stage (bps)"
TS_PARAMS_SCORE_MEMORY_USAGE_B              = "Total memory usage over the execution period (b)"
TS_PARAMS_SCORE_ASM_OPS                     = "Accumulated number of assmebly-code isntructions"
TS_PARAMS_SCORE_MIN_FREQUENCY_HZ            = "Minimum frequency at which the system needs to run (Hz)"
TS_PARAMS_SCORE_ENERGY_J                    = "Total energy consumption of the task (J)"
TS_PARAMS_SCORE_ENERGY_W                    = "Total power consumption during the task (W)"

TS_PARAMS_SELECTED_SCORES                   = "Scores selected for total score"
TS_PARAMS_SCORE_TOTAL                       = "Total score (product of selected scores)"

TS_PARAMS_METHOD                            = "Acquisition method"


TS_PARAMS_GROUND_TRUTH                      = "Ground truth series"

TS_PARAMS_DET_PEAK_Q_SLOPE_DT_S             = "Q-wave max slope time"
TS_PARAMS_DET_PEAK_Q_SLOPE_DX_REL           = "Q-wave relative amplitude change in dt_s"
TS_PARAMS_DET_PEAK_R_SLOPE_DX_REL           = "R-wave relative amplitude change in dt_s"

TS_PARAMS_CLASS_QRS_R_AMPL_MIN              = "QRS class, R-peak minimum amplitude relative"
TS_PARAMS_CLASS_QRS_OFFSET_TOLERANCE_S      = "QRS class, peak offset tolerance (s)"

TS_PARAMS_DET_FR_R_PEAK_DT_S                = "FR R-peak detection dt"
TS_PARAMS_DET_FR_R_PEAK_DXDT_TH             = "FR R-peak detection dxdt threshold"
TS_PARAMS_DET_FR_R_PEAK_AVG_TH              = "FR R-peak detection Average threshold"
TS_PARAMS_DET_FR_R_PEAK_AVG_SCOPE           = "FR R-peak detection Average scope"

TS_PARAMS_DET_LC_R_PEAK_LENGTH_N            = "LC R-peak detection Xings count"
TS_PARAMS_DET_LC_R_PEAK_DT_S                = "LC R-peak detection Xings interval (s)"

TS_PARAMS_DET_R_PEAK_SCORING_TOLERANCE      = "R-peak detection scoring tolerance"
TS_PARAMS_DET_R_PEAK_SCORING_STRAT          = "R-peak detection scoring strategy"
TS_PARAMS_DET_R_PEAK_SCORE_DET_RATE         = "R-peak detection score detection rate"
TS_PARAMS_DET_R_PEAK_SCORE_FA_RATE_INV      = "R-peak detection score false alarm rate (1-far)"

TS_PARAMS_DISTR_AIDIS_LIST                  = "List of evaluated AIDIs"
TS_PARAMS_DISTR_AIDIS_MARGIN                = "Margin to discard AIDI"

TS_PARAMS_ADC_EPC_J                         = "Energy per conversion (J)"

TS_PARAMS_LC_LVLS                           = "LC levels"
TS_PARAMS_LC_LVL_W_FRACT                    = "LC level width by fraction"
TS_PARAMS_LC_LVL_W_B                        = "LC level width by bits"
TS_PARAMS_LC_STRAT                          = "LC strategy"
TS_PARAMS_LC_FREQ_LIM                       = "LC ADC datarate limit"
TS_PARAMS_LC_ACQ_AMP_B                      = "LC Acquisition word size of Amplitude"
TS_PARAMS_LC_ACQ_DIR_B                      = "LC Acquisition word size of Direction"
TS_PARAMS_LC_ACQ_TIME_B                     = "LC Acquisition word size of Time"
TS_PARAMS_TIMER_F_HZ                     = "LC Timer frequency (Hz)"
TS_PARAMS_LC_ACQ_AMP_STRAT                  = "LC Acquisition strategy amplitude"
TS_PARAMS_LC_ACQ_DIR_STRAT                  = "LC Acquisition strategy direction"
TS_PARAMS_LC_ACQ_TIME_STRAT                 = "LC Acquisition strategy time"
TS_PARAMS_LC_ACQ_F_HZ                       = "LC Acquisition ~ frequency (Hz)"
TS_PARAMS_LC_REC_TIME                       = "LC Reconstruction (time only)"
TS_PARAMS_LC_REC_METHOD                     = "LC reconstruction method"

TS_PARAMS_AMPL_RANGE                        = "Amplitude range"
TS_PARAMS_NOISE_DROP_RATE_DBPDEC            = "Noise drop rate (dB/dec)"
TS_PARAMS_NOISE_DC_COMP                     = "DC component (???)"

TS_PARAMS_INFODATA                          = "Informative data"
TS_PARAMS_ANN                               = "Annotations (for informative data)"




TIME_FORMAT_DIFF_TM_N   = 'Number of timer periods after previous sample'
TIME_FORMAT_DIFF_FS_N   = 'Number of sampling periods after previous sample'
TIME_FORMAT_DIFF_S      = 'Time (s) after previous sample'
TIME_FORMAT_ABS_S       = 'Absolute time (s) of this sample'

METHODS_FRADC       = "FR-ADC"
METHODS_LCADC       = "LC-ADC"
METHODS_LCSUB       = "LCsubs"