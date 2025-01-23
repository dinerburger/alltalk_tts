"""
Version: 1.0
Last Updated: 2024

Implementation Notes:
- All core system variables must be maintained
- All functions prefixed with "DONT CHANGE" must remain unmodified
- Code additions should be placed between the marked sections in each function
- Debug messages use self.print_message() with appropriate message types
- All functions must implement self.debug_func_entry() for trace logging

Note: Text between "↑↑↑ Keep everything above this line ↑↑↑" and "↓↓↓ Keep everything below this line ↓↓↓"
markers **TYPICALLY** must remain unchanged as it contains critical system integration code.

Note: You can add new functions, just DONT remove the functions that are already there, even if they 
are doing nothing as `tts_server.py` will still look for their existance and fail if they are missing.
"""

########################################
# Default imports # Do not change this #
########################################
import os
import gc
import sys
import glob
import json
import time
import inspect
import torch
import logging
from pathlib import Path
from fastapi import (HTTPException)
logging.disable(logging.WARNING)

# Confguration file management for confignew.json 
try:
    from .config import AlltalkConfig, AlltalkTTSEnginesConfig, AlltalkNewEnginesConfig # TGWUI import
except ImportError:
    from config import AlltalkConfig, AlltalkTTSEnginesConfig, AlltalkNewEnginesConfig # Standalone import

def initialize_configs():
    """Initialize all configuration instances"""
    config = AlltalkConfig.get_instance()
    tts_engines_config = AlltalkTTSEnginesConfig.get_instance()
    new_engines_config = AlltalkNewEnginesConfig.get_instance()
    return config, tts_engines_config, new_engines_config

# Load in the central config management
config, tts_engines_config, new_engines_config = initialize_configs()

######################################################
# Get Pytorch & Python versions # Do not change this #
######################################################
pytorch_version = torch.__version__
cuda_version = torch.version.cuda
major, minor, micro = sys.version_info[:3]
python_version = f"{major}.{minor}.{micro}"

############################################
# DeepSpeed imports # POSSIBLY change this #
############################################
"""
If the new TTS engine you are importing doesnt support DeepSpeed, then
you can simply change `model_supports_deepspeed_true_or_false` to `False`
This will be much faster when starting up the engine. This is seperate
from what is stored in your `model_settings.json` file.
"""
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ↓↓↓ MODIFY THIS LINE ↓↓↓
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

model_supports_deepspeed_true_or_false = False

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# ↑↑↑ MODIFY THIS LINE ↑↑↑
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

if model_supports_deepspeed_true_or_false:
    try:
        import deepspeed
        deepspeed_available = True
    except ImportError:
        deepspeed_available = False
        pass

#######################################################
# TTS Engine-Specific Imports and Setup # Change this #
#######################################################
"""
This section is for TTS engine specific imports and global variable setup.

Guidelines:
1. Import all required modules for your TTS engine
2. Handle import errors with appropriate error messages
3. Set up any global variables or configurations needed by your engine
4. Use try/except blocks to gracefully handle missing dependencies
5. Ensure all error messages follow the AllTalk format for print messages.
   At this stage the `def print_messages` is not available, so please use
   standard `print("[AllTalk ENG] some message here")` messages.

Example structure:

try:
    # Import your TTS engine's required modules
    from your_tts_engine import required_modules
except ModuleNotFoundError:
    # Handle missing dependencies with helpful error messages
    print("Missing required modules. Please install...")
    raise

"""
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    import tempfile
    from pydub import AudioSegment, silence
    import torchaudio
    import numpy as np
    import re
    from torch import Tensor
    from typing import Tuple, Dict, List
    import soundfile as sf
    from huggingface_hub import snapshot_download, get_paths_info
except ModuleNotFoundError:
    brand = "[AllTalk ENG]"
    print(f"{brand} \033[91mError\033[0m Could not find the Llasa TTS modules.")
    print(f"{brand} \033[91mError\033[0m Please re-install the requirements.")
    raise


#########################################################
# Class setup # Change the relevant functions as needed #
#########################################################
class tts_class:
    """
    TTS Engine Implementation Class
    
    This class provides the interface between `tts_server.py` and whatever TTS engine you install.
    It handles model loading, voice management, and TTS generation in both streaming
    and non-streaming modes. Streaming will only be supported if the underlying TTS engine
    actually supports streaming.

    Key Responsibilities:
    1. Model Management:
       - Loading/unloading models
       - Managing model state between CPU and GPU
       - Handling DeepSpeed integration

    2. Voice Management:
       - Managing voice samples or model files

    3. TTS Generation:
       - Converting text to speech
       - Supporting streaming output (If the engine supports it)
       - Managing generation parameters

    4. System Integration:
       - Implementing standard AllTalk interfaces
       - Managing engine state and configuration
       - Handling resource allocation
    """

    ###############################################
    # Central print function # Do not change this #
    ###############################################
    def print_message(self, message, message_type="standard", component="ENG"):
        """
        Centralized print function for messages. Use this for print output to console.
        As this is the model Engine, all `component` printouts are set to ENG as default.
        
        Args:
            message (str): The message to print
            message_type (str): Type of message (standard/warning/error/debug_*/debug)
            component (str): Component identifier (TTS/ENG/GEN/API/etc.)

        Example Use:
            self.print_message("This is a standard print out mesage to a user)
            self.print_message("This is a debug_tts message, message_type="debug_tts")
            self.print_message("This is an error message to a user, message_type="error")
            self.print_message("This is an warning message to a user, message_type="warning")

        Debug Types:
            debug_func: Tracks function entry
            debug_tts: Enable TTS process debugging
            debug_tts_variables: Enable variable state debugging

        WARNING: This is a core system function. Do not modify its implementation
        as it provides standardized version reporting across all engines.
        """
        # ANSI color codes
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        YELLOW = "\033[93m"  
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m" 
        prefix = f"[{config.branding}{component}] "
        if message_type.startswith("debug_"):
            debug_flag = getattr(config.debugging, message_type, False)
            if not debug_flag:
                return
            if message_type == "debug_func" and "Function entry:" in message:
                message_parts = message.split("Function entry:", 1)
                print(f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} Function entry:{GREEN}{message_parts[1]}{RESET} in model_engine")
            else:
                print(f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} {message}")
        elif message_type == "debug":
            print(f"{prefix}{BLUE}Debug{RESET} {message}")  
        elif message_type == "warning":
            print(f"{prefix}{YELLOW}Warning{RESET} {message}")  
        elif message_type == "error":
            print(f"{prefix}{RED}Error{RESET} {message}")  
        else:
            print(f"{prefix}{message}")

    def debug_func_entry(self):
        """Log function entry if debug_func is enabled."""
        if config.debugging.debug_func:
            current_func = inspect.currentframe().f_back.f_code.co_name
            self.print_message(f"Function entry: {current_func}", "debug_func")

    #############################################
    # Script initalisation # Do not change this #
    #############################################
    def __init__(self):
        """
        Initialize the TTS engine instance.
        
        WARNING: This class requires specific variables to interface with AllTalk's main system (tts_server.py).
        Do not remove or rename any of the predefined variables as they are required for proper system integration.
        
        Required System Interface Variables:

        1. Core System Variables (DO NOT MODIFY):
           Base Configuration:
           - self.this_dir: Engine directory path (where this script is located)
           - self.main_dir: AllTalk root directory
           - self.device: Processing device ("cuda" or "cpu")
           - self.cuda_is_available: Whether GPU/CUDA is available
           
           State Tracking:
           - self.tts_generating_lock: Prevents concurrent generation requests 
           - self.tts_stop_generation: Signals generation stop request
           - self.tts_narrator_generatingtts: Tracks narrator mode for optimization
           - self.model: Active TTS model instance
           - self.is_tts_model_loaded: Whether a model is currently loaded
           - self.current_model_loaded: Name of currently loaded model
           - self.available_models: List of models found by scan_models_folder
           - self.setup_has_run: Tracks if setup() has completed
        
        2. Engine Configuration Variables (DO NOT MODIFY):
           - self.engines_available: List of all available TTS engines
           - self.engine_loaded: Currently selected TTS engine
           - self.selected_model: Currently selected model name
        
        3. Model Settings (SET VIA model_settings.json):
           Capability Flags:
           - self.audio_format: Output audio format (wav, mp3, etc.)
           - self.deepspeed_capable: DeepSpeed acceleration support
           - self.generationspeed_capable: Speed adjustment support
           - self.languages_capable: Multi-language support
           - self.lowvram_capable: Low VRAM mode support
           - self.multimodel_capable: Multiple model support
           - self.repetitionpenalty_capable: Repetition penalty support
           - self.streaming_capable: Audio streaming support
           - self.temperature_capable: Temperature adjustment support
           - self.multivoice_capable: Multiple voice support
           - self.pitch_capable: Pitch adjustment support
           
           Engine Settings:
           - self.def_character_voice: Default character voice
           - self.def_narrator_voice: Default narrator voice
           - self.deepspeed_enabled: DeepSpeed status
           - self.engine_installed: Engine installation status
           - self.generationspeed_set: Current speed setting
           - self.lowvram_enabled: Low VRAM mode status
           - self.repetitionpenalty_set: Current repetition penalty
           - self.temperature_set: Current temperature setting
           - self.pitch_set: Current pitch setting
           
           OpenAI Voice Mappings:
           - self.openai_alloy: Alloy voice mapping
           - self.openai_echo: Echo voice mapping
           - self.openai_fable: Fable voice mapping
           - self.openai_nova: Nova voice mapping
           - self.openai_onyx: Onyx voice mapping
           - self.openai_shimmer: Shimmer voice mapping
        
        Integration Requirements:
        - All variables must be present even if unused by your engine
        - Capability flags should accurately reflect engine features
        - Settings should have sensible defaults even if not used
        - OpenAI mappings should be set even if not supporting OpenAI compatibility
        
        Note: Variables marked (DO NOT MODIFY) are critical system integration points.
        Other variables should be configured through their respective JSON files or
        you can add new central variables in the section provided down below.
        """
        # DO NOT MODIFY - Sets up the base variables required for any tts engine #
        self.this_dir = Path(__file__).parent.resolve()
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_is_available = torch.cuda.is_available()
        self.tts_generating_lock = False
        self.tts_stop_generation = False
        self.tts_narrator_generatingtts = False
        self.model = None
        self.is_tts_model_loaded = False
        self.current_model_loaded = None
        self.available_models = None
        self.setup_has_run = False
        self.engines_available = tts_engines_config.get_engine_names_available()
        self.engine_loaded = tts_engines_config.engine_loaded
        self.selected_model = tts_engines_config.selected_model
        self.target_sample_rate = 16000
        self.cross_fade_duration = 0.15

        # DO NOT MODIFY - Load in the current TTS Engines model_settings.json file
        with open(os.path.join(self.this_dir, "model_settings.json"), "r") as f:
            model_settings_file = json.load(f)
    
        # DO NOT MODIFY - Model details from model_settings.json
        self.manufacturer_name = model_settings_file["model_details"]["manufacturer_name"]
        self.manufacturer_website = model_settings_file["model_details"]["manufacturer_website"]
        
        # DO NOT MODIFY - Model capabilities from model_settings.json
        self.audio_format = model_settings_file["model_capabilties"]["audio_format"]
        self.deepspeed_capable = model_settings_file["model_capabilties"]["deepspeed_capable"]
        self.deepspeed_available = 'deepspeed' in globals()
        self.generationspeed_capable = model_settings_file["model_capabilties"]["generationspeed_capable"]
        self.languages_capable = model_settings_file["model_capabilties"]["languages_capable"]
        self.lowvram_capable = model_settings_file["model_capabilties"]["lowvram_capable"]
        self.multimodel_capable = model_settings_file["model_capabilties"]["multimodel_capable"]
        self.repetitionpenalty_capable = model_settings_file["model_capabilties"]["repetitionpenalty_capable"]
        self.streaming_capable = model_settings_file["model_capabilties"]["streaming_capable"]
        self.temperature_capable = model_settings_file["model_capabilties"]["temperature_capable"]
        self.multivoice_capable = model_settings_file["model_capabilties"]["multivoice_capable"]
        self.pitch_capable = model_settings_file["model_capabilties"]["pitch_capable"]
        
        # DO NOT MODIFY - Engine settings from model_settings.json
        self.def_character_voice = model_settings_file["settings"]["def_character_voice"]
        self.def_narrator_voice = model_settings_file["settings"]["def_narrator_voice"]
        self.deepspeed_enabled = model_settings_file["settings"]["deepspeed_enabled"]
        self.engine_installed = model_settings_file["settings"]["engine_installed"]
        self.generationspeed_set = model_settings_file["settings"]["generationspeed_set"]
        self.lowvram_enabled = model_settings_file["settings"]["lowvram_enabled"]
        self.lowvram_enabled = False if not torch.cuda.is_available() else self.lowvram_enabled
        self.repetitionpenalty_set = model_settings_file["settings"]["repetitionpenalty_set"]
        self.temperature_set = model_settings_file["settings"]["temperature_set"]
        self.pitch_set = model_settings_file["settings"]["pitch_set"]
        
        # DO NOT MODIFY - OpenAI voice mappings from model_settings.json
        self.openai_alloy = model_settings_file["openai_voices"]["alloy"]
        self.openai_echo = model_settings_file["openai_voices"]["echo"]
        self.openai_fable = model_settings_file["openai_voices"]["fable"]
        self.openai_nova = model_settings_file["openai_voices"]["nova"]
        self.openai_onyx = model_settings_file["openai_voices"]["onyx"]
        self.openai_shimmer = model_settings_file["openai_voices"]["shimmer"]

        self.branding = "LLaSA"

        """
        Below is the name of the folder that will be created-used under `/models/{folder}`
        And is further used by the base functions of this script. Change the name stored in
        `self.model_folder_name` to the model folder name you will be using. Ensure to use
        the correct CAPS/Non-CAPS spelling as Linux OS is CAPS specific on folder names.
        """
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ MODIFY THIS LINE ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        self.model_folder_name = "llasa"
         
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ MODIFY THIS LINE ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Add your own central `self.myvariable` variables in here if needed for your engine ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        """
        If you need globally accessable variables of your own for your own purposes, you can put
        them in here as self.myvariable = "whatever".

        """
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Add your own central `self.myvariable` variables in here if needed for your engine ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # DO NOT MODIFY - log the function call to this function
        self.debug_func_entry() 

    #####################################################
    # Printout engine loading bits # Do not change this #
    #####################################################
    def printout_versions(self):
        """
        Print Python, DeepSpeed, Pytorch and CUDA version on start-up.
        
        WARNING: This is a core system function. Do not modify its implementation
        as it provides standardized version reporting across all engines.
        """
        self.debug_func_entry()
        if not model_supports_deepspeed_true_or_false:
            self.print_message(f"\033[92mDeepSpeed version :\033[93m Not supported on {self.model_folder_name}\033[0m", message_type="standard")
        else:
            if deepspeed_available:
                self.print_message("\033[92mDeepSpeed version :\033[93m " + str(deepspeed.__version__) + "\033[0m", message_type="standard")
            else:
                self.print_message("\033[92mDeepSpeed version :\033[93m Not available\033[0m", message_type="standard")
        self.print_message(f"\033[92mPython Version    :\033[93m {python_version}\033[0m", message_type="standard")
        self.print_message(f"\033[92mPyTorch Version   :\033[93m {pytorch_version}\033[0m", message_type="standard")
        if cuda_version is None:
            self.print_message("\033[92mCUDA Version      :\033[91m Not available\033[0m", message_type="standard")
        else:
            self.print_message(f"\033[92mCUDA Version      :\033[93m {cuda_version}\033[0m", message_type="standard")
            
        self.print_message("", message_type="standard")
        return

    ################################################################
    # Handle low VRAM change between CUDA/CPU # Do not change this #
    ################################################################
    async def handle_lowvram_change(self):
        """
        Manage model location between CPU and GPU memory for low VRAM operation.
        
        This function handles the movement of models between CPU and GPU memory
        to support systems with limited VRAM. It's called automatically during
        generation when low VRAM mode is enabled.
        
        Operation:
        1. Checks CUDA availability
        2. Moves model between devices based on current location:
           - GPU (cuda) -> CPU
           - CPU -> GPU (cuda)
        3. Manages CUDA cache to optimize memory usage
        
        States Affected:
        - self.device: Updated to reflect current processing device
        - self.model.device: Model's current memory location
        
        Requirements:
        - CUDA must be available for GPU operations
        - Model must be loaded (self.model is not None)
        - lowvram_enabled must be `True` in the `model_settings.json` file
        
        Note: This function is only called when self.lowvram_enabled is True
        meaning the engine does or doesnt support the call, hence if its not
        True, then this function would never be called anyway, so doesnt need
        changing.
        """
        self.debug_func_entry()
        
        # Initial validation
        if not self.is_tts_model_loaded:
            self.print_message("No model is currently loaded. Please select a model to load.", message_type="error")
            raise HTTPException(status_code=400, detail="No model is currently loaded. Please select a model to load.")
                       
        if torch.cuda.is_available():
            if self.device == "cuda":
                self.print_message("Moving model to CPU", message_type="debug_tts")
                self.device = "cpu"
                self.model.to(self.device)
                torch.cuda.empty_cache()
                gc.collect()
            else:
                self.device = "cuda"
                self.print_message("Moving model to GPU", message_type="debug_tts")
                self.model.to(self.device)
                gc.collect()

    ################################################
    # Handle DeepSpeed change # Do not change this #
    ################################################
    async def handle_deepspeed_change(self, value):
        """
        Handle enabling/disabling of DeepSpeed acceleration.
        
        This function manages the process of reloading the model with or without 
        DeepSpeed acceleration. DeepSpeed can significantly improve performance on 
        supported hardware.
        
        Args:
            value (bool): True to enable DeepSpeed, False to disable
            
        Operation:
        1. Unloads current model
        2. Updates DeepSpeed settings
        3. Reloads model with new configuration
        
        States Affected:
        - self.deepspeed_enabled: Updated to reflect new state
        - self.model: Reloaded with new configuration
        
        Returns:
            bool: The new DeepSpeed state (same as input value)
        
        Note: DeepSpeed must be installed and available in the system for 
        this functionality to work. `deepspeed_capable` must be set `True`
        in the `model_settings.json` file
        """
        self.debug_func_entry()
        # Initial validation        
        if not self.is_tts_model_loaded:
            self.print_message("No model is currently loaded. Please select a model to load.", message_type="error")
            raise HTTPException(status_code=400, detail="No model is currently loaded. Please select a model to load.")
        
        if value:
            self.print_message("\033[93mDeepSpeed Activating\033[0m", message_type="standard")
            await self.unload_model()
            self.deepspeed_enabled = True
            await self.setup()
        else:
            self.print_message("\033[93mDeepSpeed De-Activating\033[0m", message_type="standard")
            self.deepspeed_enabled = False
            await self.unload_model()
            await self.setup()
        return value


    ################################################################
    # Unload models from VRAM/RAM if possible # Do not change this #
    ################################################################
    async def unload_model(self):
        """
        Unload the current model and free associated resources.
        
        This function handles the cleanup of model resources, including:
        1. Setting model loaded flag to False
        2. Deleting the model instance
        3. Clearing CUDA cache if available
        
        Operation:
        1. Updates model loading status
        2. Logs unloading process if a model is loaded
        3. Removes model from memory
        4. Cleans up CUDA cache if using GPU
        
        States Affected:
        - self.is_tts_model_loaded: Set to False
        - self.model: Set to None after unloading
        """
        self.debug_func_entry()
        
        self.is_tts_model_loaded = False
        if not self.current_model_loaded == None:
            self.print_message("Unloading model", message_type="debug_tts")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'codec_model'):
            del self.codec_model        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    ############################################################
    # On start-up, perform these actions # Change as necessary #
    ############################################################
    async def setup(self):
        """
        Initialize the TTS engine and load initial model configuration.
        
        This function is called during system startup and handles:
        1. Version information display
        2. Model scanning and availability check
        3. Initial model loading if specified
        
        The setup sequence ensures:
        - Proper version reporting
        - Model availability verification
        - Graceful handling of missing models
        - Correct initial model loading state
        
        States Set:
        - self.available_models: Updated with found models
        - self.current_model_loaded: Set to loaded model name or None
        - self.setup_has_run: Set True when complete
        
        Returns:
            None
            
        Note: Custom initialization code should be placed between the marked sections.
        """
        self.debug_func_entry()
        self.print_message("Initializing TTS engine", message_type="debug_tts")
        self.printout_versions()
        self.available_models = self.scan_models_folder()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        if self.selected_model:
            tts_model = f"{self.selected_model}"
            if tts_model in self.available_models:
                self.print_message(f"Loading selected model: {tts_model}", message_type="debug_tts")
                await self.handle_tts_method_change(tts_model)
                self.current_model_loaded = tts_model
                
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓                  
            else:
                self.current_model_loaded = "No Models Available"
                self.print_message(f"Selected model '{self.selected_model}' not found in models folder.", message_type="error")
                self.print_message(f"Please download a model or select a different model file.", message_type="error")
        self.setup_has_run = True


    ###########################################################################
    # Scan your models folder for models OR voice files # Change as necessary #
    ###########################################################################
    def scan_models_folder(self):
        """
        Scan for available TTS models in the models directory.
        
        This function searches the models directory for valid TTS model installations.
        Each model must contain all required files to be considered valid.
        
        Required Files for Each Model:
        - Whatever your TTS engine needs/supports
        
        Operation:
        1. Scans the models/{folder} directory
        2. Checks each subfolder for required files
        3. Registers valid models
        
        States Affected:
        - self.available_models: Updated with found models
        
        Returns:
            dict: Dictionary of available models in format:
                 {model_identifier: engine_type}
        
        Note: If no valid models are found, returns {"No Models Available": "TTS Engine Name"}
        """
        self.debug_func_entry()
        
        models_folder = self.main_dir / "models" / self.model_folder_name
        self.available_models = {}
        
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑  

        self.print_message(f"models_folder is {models_folder}", message_type="debug")


        # Add immediate check for models folder existence
        if not models_folder.exists():
            self.print_message(f"Models folder does not exist: {models_folder}", message_type="warning")
            self.print_message(f"Please use the Gradio inteface to download/select a model.", message_type="warning")
            self.available_models = {'No Models Available': 'llasa'}
            return self.available_models


        # Check basic files needed for any model
        common_required_files = [
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]

        model_name = "DO SOMETHING HERE"

        found_valid_model = False
        for subfolder in models_folder.iterdir():
            if subfolder.is_dir():
                model_name = subfolder.name
                self.print_message(f"model_name is {model_name}", message_type="debug")

                # Check for common required files
                if all(subfolder.joinpath(file).exists() for file in common_required_files):
                    # Look for any safetensors files
                    safetensor_files = list(subfolder.glob("*.safetensors"))
                    if safetensor_files:
                        self.available_models[f"llasa - {model_name}"] = "llasa"
                        found_valid_model = True
                        self.print_message(f"self.available_models is: {self.available_models}", message_type="debug")
                    else:
                        self.print_message(f"Model folder '{model_name}' has no .safetensors files", message_type="warning")
                else:
                    self.print_message(f"Model folder '{model_name}' is missing common required files", message_type="warning")

        if not found_valid_model:
            self.available_models["No Models Found"] = "No Models Found" # Return a list with {'No Models Found'} if there are no models found.


        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓              
        return self.available_models

    ################################################################
    # Scan your voice folder for voice files # Change as necessary #
    ################################################################
    def voices_file_list(self):
        try:
            voices = []
            # Function to check if a wav file has a corresponding reference text file
            def has_reference_text(wav_path: Path):
                text_path = wav_path.with_suffix('.reference.txt')
                return text_path.exists()
            
            directory = self.main_dir / "voices"
            
            # Step 1: Add .wav files in the main "voices" directory to the list (only if they have matching .reference.txt)
            for f in directory.glob("*.wav"):
                if has_reference_text(f):
                    voices.append(f.name)
                else:
                    self.print_message(f"Warning: {f.name} does not have a matching reference text file", message_type="warning")
            
            # Remove "voices/" from the list if it somehow got added
            voices = [v for v in voices if v != "voices/"]
                        
            if not voices:
                return ["No Voices Found"] 
            return voices 
        except Exception as e:
            self.print_message(f"Voices/Voice Models not found. Cannot load a list of voices", message_type="error")
            return ["No Voices Found"]
    
    ############################################
    # Load in your model # Change as necessary #
    ############################################
    async def load_model(self, model_name: str):
        """
        Load a model using the your TTS API interface.
               
        Args:
            model_name (str): Name of the model to load
            
        Operation:
        1. Validates model availability
        2. Constructs model and config paths
        3. Initializes model using TTS API
        4. Moves model to appropriate device (CPU/GPU)
        
        States Affected:
        - self.model: Updated with loaded model
        - self.is_tts_model_loaded: Set to True on success
        
        Returns:
            The loaded model instance
            
        Raises:
            HTTPException: If no models are available to load
        """
        self.debug_func_entry()
        if "No Models Available" in self.available_models:
            self.print_message("No models for this TTS engine were found to load", message_type="error")
            return
        model_path = self.main_dir / "models" / self.model_folder_name / model_name.replace("llasa - ", "") # You may need to edit/modfy depending on how your TTS engine works with models
        llasa_base_path = self.main_dir / "models" / "llasa_base"
        codec_model_path = self.main_dir / "models" / "llasa_base" / "xcodec2"
        self.print_message(f"Loading model from: {model_path}", message_type="debug_tts")
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑        
        
        if not model_path.exists():
            self.print_message(f"Model directory not found: {model_path}", message_type="error")
            self.print_message(f"Please download a llasa model file in the Gradio interface section for the Parler engine.", message_type="error")
            raise HTTPException(status_code=404, detail=f"Model directory not found: {model_path}")
       
        # Get the appropriate dtype
        dtype = torch.bfloat16 if self.cuda_is_available else torch.float32
        
        # Print dtype information
        dtype_name = str(dtype).split('.')[-1]
        self.print_message(f"Loading model with dtype: {dtype_name}", message_type="standard")
        
        try:
            snapshot_download("HKUSTAudio/xcodec2", local_dir=str(codec_model_path), ignore_patterns=["__pycache__", "ckpt", "*.ckpt", "*.pyc"])
            if not str(llasa_base_path) in sys.path:
                sys.path.append(str(llasa_base_path))
            if not str(codec_model_path) in sys.path:
                sys.path.append(str(codec_model_path))
            from xcodec2.modeling_xcodec2 import XCodec2Model
            self.codec_model = XCodec2Model.from_pretrained(str(codec_model_path),
                                                            trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                device_map=self.device
            )
            if self.cuda_is_available:
                self.codec_model.eval().cuda()
            else:
                self.codec_model.eval()

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2
                self.print_message(f"GPU Memory Used: {memory_used:.2f} MB", message_type="standard")
            
        except Exception as e:
            self.print_message(f"Error loading model: {str(e)}", message_type="error")
            return None
        
      
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.is_tts_model_loaded = True
        return self.model

    async def handle_tts_method_change(self, tts_method):
        """
        Handle switching between different TTS models/voices.
        
        This function manages actual model loading process.
        
        Args:
            tts_method (str): Format "type - modelname" where type is either
        
        Operation:
        1. Validates model availability
        2. Unloads current model if any
        3. Parses method string to determine loader type
        4. Calls appropriate model loader
        5. Updates current model tracking
        
        States Affected:
        - self.current_model_loaded: Updated to new model identifier
        - self.model: Updated with newly loaded model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        
        Timing:
            Records and reports model loading time
        """
        self.debug_func_entry()
        
        # Track loading time
        generate_start_time = time.time()
        
        # Validate model availability
        if "No Models Available" in self.available_models:
            self.print_message("No models for this TTS engine were found to load", message_type="error")
            return False

        # Unload current model
        await self.unload_model()
        
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Load the new model
        try:
            await self.load_model(tts_method)
            self.current_model_loaded = tts_method
        except Exception as e:
            self.print_message(f"Error loading model: {str(e)}", message_type="error")
            return False

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        # Report loading time
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        self.print_message(f"\033[94mModel Loadtime: \033[93m{generate_elapsed_time:.2f}\033[94m seconds\033[0m")
        return True

    def chunk_text(self, text, max_chars=300) -> List[str]:
        """
        Splits the input text into chunks, each with a maximum number of characters.
        """
        chunks = []
        current_chunk = ""
        # Split the text into sentences based on punctuation followed by whitespace
        sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

        for sentence in sentences:
            if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
                current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def ids_to_speech_tokens(self, speech_ids: List[str]):
        speech_tokens_str = []
        for speech_id in speech_ids:
            speech_tokens_str.append(f"<|s_{speech_id}|>")
        return speech_tokens_str

    def extract_speech_ids(self, speech_tokens_str: List[str]):
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith('<|s_') and token_str.endswith('|>'):
                num_str = token_str[4:-2]

                num = int(num_str)
                speech_ids.append(num)
            else:
                self.print_message(f"Unexpected token: {token_str}", message_type="warning")
        return speech_ids

    async def preprocess_ref_audio_text(self, ref_audio_path: str, ref_text: str, clip_short=True):
        self.print_message(f"Converting audio...", message_type="debug")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            aseg = AudioSegment.from_file(ref_audio_path)

            if clip_short:
                # 1. try to find long silence for clipping
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                        self.print_message(f"Audio is over 15s, clipping short. (1)", message_type="debug")
                        break
                    non_silent_wave += non_silent_seg

                # 2. try to find short silence for clipping if 1. failed
                if len(non_silent_wave) > 15000:
                    non_silent_segs = silence.split_on_silence(
                        aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000
                    )
                    non_silent_wave = AudioSegment.silent(duration=0)
                    for non_silent_seg in non_silent_segs:
                        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                            self.print_message(f"Audio is over 15s, clipping short. (2)", message_type="debug")
                            break
                        non_silent_wave += non_silent_seg

                aseg = non_silent_wave

                # 3. if no proper silence found for clipping
                if len(aseg) > 15000:
                    aseg = aseg[:15000]
                    self.print_message(f"Audio is over 15s, clipping short. (3)", message_type="debug")

            aseg.export(f.name, format="wav")
            ref_audio = f.name

        # Ensure ref_text ends with a proper sentence-ending punctuation
        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "

        return ref_audio, ref_text

    async def infer_process(
        self,
        ref_audio,
        ref_text,
        gen_text,
        cross_fade_duration=0.15,
        temperature=0.8
    ):
        """Process text and prepare for batch inference"""
        # Split the input text into batches
        audio, sr = torchaudio.load(str(ref_audio))
        gen_text_batches = self.chunk_text(gen_text)
        
        for i, gen_text_batch in enumerate(gen_text_batches):
            self.print_message(f"gen_text {gen_text_batch}", message_type="debug")

        self.print_message(f"Generating audio in {len(gen_text_batches)} batches...", message_type="debug")
        
        return await self.infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            cross_fade_duration=cross_fade_duration,
            temperature=temperature
        )

    async def infer_batch_process(
        self,
        ref_audio: Tuple[Tensor, int],
        ref_text: str,
        gen_text_batches: List[str],
        cross_fade_duration=0.15,
        temperature=0.8
    ):
        """Process batches for inference"""
        device = self.device
        audio, sr = ref_audio
        
        # Always process initial audio in float32
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)

        generated_waves = []

        if len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "
            
        for gen_text in gen_text_batches:
            # Prepare the text
            input_text = ref_text + gen_text
            #TTS start!
            with torch.no_grad():
                # Encode the prompt wav
                vq_code_prompt = self.codec_model.encode_code(input_waveform=audio)
                vq_code_prompt = vq_code_prompt[0,0,:]
                # Convert int 12345 to token <|s_12345|>
                speech_ids_prefix = self.ids_to_speech_tokens(vq_code_prompt)

                formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

                # Tokenize the text and the speech prefix
                chat = [
                    {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                    {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
                ]

                input_ids = self.tokenizer.apply_chat_template(
                    chat, 
                    tokenize=True, 
                    return_tensors='pt', 
                    continue_final_message=True
                )
                if self.cuda_is_available:
                    input_ids = input_ids.to('cuda')
                speech_end_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

                # Generate the speech autoregressively
                outputs = self.model.generate(
                    input_ids,
                    max_length=2048,  # We trained our model with a max length of 2048
                    eos_token_id= speech_end_id ,
                    do_sample=True,
                    top_p=1,           
                    temperature=temperature
                )
                # Extract the speech tokens
                generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]

                speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   

                # Convert  token <|s_23456|> to int 23456 
                speech_tokens = self.extract_speech_ids(speech_tokens)

                if self.cuda_is_available:
                    speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
                else:
                    speech_tokens = torch.tensor(speech_tokens).unsqueeze(0).unsqueeze(0)

                # Decode the speech tokens to speech waveform
                gen_wav = self.codec_model.decode_code(speech_tokens) 

                # if only need the generated part
                gen_wav = gen_wav[:,:,audio.shape[1]:]
                generated_waves.append(gen_wav[0, 0, :].cpu().numpy())

        # Combine all generated waves with cross-fading
        if cross_fade_duration <= 0:
            final_wave = np.concatenate(generated_waves)
        else:
            final_wave = generated_waves[0]
            for i in range(1, len(generated_waves)):
                prev_wave = final_wave
                next_wave = generated_waves[i]

                # Calculate cross-fade samples
                cross_fade_samples = int(cross_fade_duration * self.target_sample_rate)
                cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                if cross_fade_samples <= 0:
                    final_wave = np.concatenate([prev_wave, next_wave])
                    continue

                # Overlapping parts
                prev_overlap = prev_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]

                # Fade out and fade in
                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)

                # Cross-faded overlap
                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                # Combine
                new_wave = np.concatenate(
                    [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                )

                final_wave = new_wave

        return final_wave, self.target_sample_rate

    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        """
        Generate speech from text using the TTS model.

        This core function handles all TTS generation, supporting both streaming and
        non-streaming output, multiple voice input types,
        and various generation parameters.

        Args:
            text (str): Text to convert to speech
            voice (str): Voice identifier (WAV/MP3 file, voice set, or latent)
            language (str): Target language code
            temperature (float): Generation temperature (0.0-1.0)
            repetition_penalty (float): Penalty for repetitive generation
            speed (float): Speech speed multiplier
            pitch (float): Voice pitch adjustment 
            output_file (str): Path for output audio file
            streaming (bool): Whether to stream audio chunks

        Returns:
            For streaming=True: Generator yielding audio chunks
            For streaming=False: None (saves to output_file)

        States Used:
            - self.model: Active TTS model
            - self.device: Current processing device
            - self.lowvram_enabled: Low VRAM mode status
            - self.current_model_loaded: Current model type
        """        
        self.debug_func_entry()
        
        # Initial validation
        if not self.is_tts_model_loaded:
            self.print_message("No TTS model loaded", message_type="error")
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")
        
        # Lock generation and track start time
        self.tts_generating_lock = True
        self.print_message("Starting TTS generation process", message_type="debug_tts")
        self.print_message(f"Generation parameters: temperature={temperature}, speed={speed}, streaming={streaming}", 
                        message_type="debug_tts_variables")
        
        # Handle low VRAM mode if needed
        if self.lowvram_enabled and self.device == "cpu":
            self.print_message("Low VRAM mode: Moving model to GPU", message_type="debug_tts")
            await self.handle_lowvram_change()
        
        generate_start_time = time.time()

        try:
            # Voice input processing
            self.print_message(f"Processing voice input: {voice}", message_type="debug_tts")

            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            # ↑↑↑ Keep everything above this line ↑↑↑
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            # BUILD ANY SETTINGS NEEED FOR TTS HERE
            # DO ANY LOGIC TESTS E.G. CHECK IF THE AUDIO FILE/MODEL EXISTS

            self.print_message("Starting non-streaming generation", message_type="debug_tts")

            # It's a direct file
            ref_audio_path = self.main_dir / "voices" / voice
                
            # Get the corresponding reference text file
            ref_text_path = ref_audio_path.with_suffix('.reference.txt')
            if not ref_text_path.exists():
                raise FileNotFoundError(f"Reference text file not found for {voice}")
                
            with open(ref_text_path, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
                
            # Process reference audio and text
            ref_audio, processed_ref_text = await self.preprocess_ref_audio_text(
                str(ref_audio_path), ref_text
            )
            
            # Generate the audio using infer_process
            final_wave, final_sample_rate = await self.infer_process(
                ref_audio,
                processed_ref_text,
                text,
                cross_fade_duration=self.cross_fade_duration,
                temperature=temperature
            )
            
            # Save the audio
            sf.write(output_file, final_wave, final_sample_rate)
            
            # PUT YOUR TTS ENGINE STANDARD GENERATION LOGIC IN HERE

            self.print_message(f"Saved audio to: {output_file}", message_type="debug_tts")

            
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        except Exception as e:
            self.print_message(f"Error generating audio: {e}", message_type="error")
            


        finally:
            # Generation complete
            generate_end_time = time.time()
            generate_elapsed_time = generate_end_time - generate_start_time
            
            # Standard output message (not debug)
            self.print_message(
                f"\033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled}\033[0m",
                message_type="standard"
            )
            
            # Handle low VRAM cleanup
            if self.lowvram_enabled and self.device == "cuda" and not self.tts_narrator_generatingtts:
                self.print_message("Low VRAM mode: Moving model back to CPU", message_type="debug_tts")
                await self.handle_lowvram_change()
                
            self.tts_generating_lock = False
