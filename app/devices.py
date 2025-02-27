import sounddevice as sd
import logging

logger = logging.getLogger("TranscriptionApp")

def list_microphones():
    try:
        devices = sd.query_devices()
        input_devices = [device for device in devices if device['max_input_channels'] > 0]
        output_devices = [device for device in devices if device['max_output_channels'] > 0]
        logger.info(f"Total Input Devices Found: {len(input_devices)}")
        logger.info(f"Total Output Devices Found: {len(output_devices)}")
        return input_devices, output_devices
    except Exception as e:
        logger.error(f"Error querying devices: {e}", exc_info=True)
        return [], []

def get_device_id(device_name, input_device=True):
    devices = list_microphones()[0] if input_device else list_microphones()[1]
    for device in devices:
        name = f"{device['name']} (ID: {device['index']})"
        if name == device_name:
            return device['index']
    return None