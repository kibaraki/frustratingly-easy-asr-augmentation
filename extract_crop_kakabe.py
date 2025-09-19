import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from pydub import AudioSegment

def parse_and_crop_data(input_dir, output_audio_dir):
 
    # Create the output directory if it doesn't exist
    os.makedirs(output_audio_dir, exist_ok=True)
    
    all_records = []
    
    # Find all XML files in the input directory
    xml_dir = input_dir + './xml/'
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    
    if not xml_files:
        print(f"Error: No XML files found in '{input_dir}'.")
        return pd.DataFrame()

    print(f"Found {len(xml_files)} XML files. Processing...")

    for xml_file_path in xml_files:
        try:
            # Set up file paths 
            base_filename = os.path.splitext(os.path.basename(xml_file_path))[0]
            wav_dir = input_dir + './audio/'
            wav_file_path = os.path.join(wav_dir, f"{base_filename}.wav")

            if not os.path.exists(wav_file_path):
                print(f"Warning: Corresponding WAV file not found for {base_filename}.xml. Skipping.")
                continue

            # Load the main audio file
            main_audio = AudioSegment.from_wav(wav_file_path)
            
            # Parse the XML file 
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # Iterate through each sentence <S> tag
            for sentence in root.findall('.//S'):
                s_id = sentence.get('id')
                
                # Extract data from XML tags 
                audio_tag = sentence.find('AUDIO')
                start_time_sec = float(audio_tag.get('start'))
                end_time_sec = float(audio_tag.get('end'))

                text = sentence.find('FORM').text
                english_translation = sentence.find('TRANSL').text
                
                # Extract and combine the glosses from each word
                gloss_parts = []
                # The structure is S -> W -> M -> TRANSL
                for word in sentence.findall('W'):
                    for morpheme in word.findall('M'):
                        gloss_tag = morpheme.find('TRANSL')
                        if gloss_tag is not None and gloss_tag.text is not None:
                            gloss_parts.append(gloss_tag.text)
                gloss = ' '.join(gloss_parts)

                # Crop the audio 
                start_time_ms = int(start_time_sec * 1000)
                end_time_ms = int(end_time_sec * 1000)
                
                audio_clip = main_audio[start_time_ms:end_time_ms]
                
                # Create a unique filename for the clip
                clip_filename = f"{base_filename}_{s_id}.wav"
                clip_output_path = os.path.join(output_audio_dir, clip_filename)
                
                # Export the cropped clip
                audio_clip.export(clip_output_path, format="wav")
                
                # Store the extracted record 
                record = {
                    'start_time': start_time_sec,
                    'end_time': end_time_sec,
                    'text': text,
                    'gloss': gloss,
                    'english_translation': english_translation,
                    'clip_filename': clip_filename
                }
                all_records.append(record)

        except Exception as e:
            print(f"Error processing file {xml_file_path}: {e}")

    print("Processing complete.")
    return pd.DataFrame(all_records)

if __name__ == '__main__':

    INPUT_DATA_DIR = './'
    
    OUTPUT_AUDIO_DIR = 'cropped_audio'
    
    # Name for the final output CSV file
    OUTPUT_CSV_FILE = 'extracted_data.csv'


    df = parse_and_crop_data(INPUT_DATA_DIR, OUTPUT_AUDIO_DIR)

    if not df.empty:

        df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        
        print(f"\nSuccessfully created {len(df)} audio clips and extracted data.")
        print(f"Data saved to '{OUTPUT_CSV_FILE}'.")
        print("\n--- Data Preview ---")
        print(df.head())
