from PIL import Image
import numpy as np
import os
import soundfile as sf
import subprocess
from omegaconf import OmegaConf, DictConfig


from moviepy.editor import VideoClip, ImageClip, TextClip

# import pdb; pdb.set_trace()


# Run the ffmpeg command to get the path
ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
# Set the environment variable
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
# Optionally, print the path for verification
print("IMAGEIO_FFMPEG_EXE set to:", os.environ["IMAGEIO_FFMPEG_EXE"])


# import moviepy.config_defaults
# magick_path = subprocess.check_output(['which', 'magick']).decode().strip()
# moviepy.config_defaults.IMAGEMAGICK_BINARY = magick_path
# print("IMAGEMAGICK_BINARY Path set to:", moviepy.config_defaults.IMAGEMAGICK_BINARY)




def create_animation_with_text(image_path1, image_path2, audio_path, output_path, image_prompt, audio_prompt):
    # set up video hyperparameter
    space_between_images = 30
    padding = 40
    space_between_text_image = 8
    fontsize = 26
    font = 'Ubuntu-Mono'
    text_method = 'caption'

    # Load images
    image1 = np.array(Image.open(image_path1).convert('RGB'))
    image2 = np.array(Image.open(image_path2).convert('RGB'))
    image1_height, image1_width, _ = image1.shape
    image2_height, image2_width, _ = image2.shape

    max_image_width = max(image1_width, image2_width)
    # Add text prompts
    text_size = [max_image_width, None]
    image_prompt = TextClip('Image prompt: ' + image_prompt, font=font, fontsize=fontsize, bg_color='white', color='black', method=text_method, size=text_size)
    audio_prompt = TextClip('Audio prompt: ' + audio_prompt, font=font, fontsize=fontsize, bg_color='white', color='black', method=text_method, size=text_size)
    image_text_height = np.array(image_prompt.get_frame(0)).shape[0]
    audio_text_height = np.array(audio_prompt.get_frame(0)).shape[0]
    # Calculate total height for the video
    total_height = image1_height + image2_height + space_between_images + 2 * padding + image_text_height + audio_text_height + 2 * space_between_text_image

    # Create a white slider image without shadow
    slider_width = 5  # Increased width
    slider_height = image1_height + image2_height + space_between_images  # Adjusted height
    slider_color = (240, 240, 240)  # White

    # Create slider without shadow
    slider = np.zeros((slider_height, slider_width, 3), dtype=np.uint8)
    slider[:, :] = slider_color  # Main slider area

    # import pdb; pdb.set_trace()
    # Load audio using soundfile
    audio_data, sample_rate = sf.read(audio_path)

    # Calculate video duration based on audio length
    video_duration = len(audio_data) / sample_rate

    # Define video dimensions
    video_width = max_image_width + 2 * padding

    # Function to generate frame at time t
    def make_frame(t):
        # Calculate slider position
        # import pdb; pdb.set_trace() 
        slider_position = int(t * (max_image_width - slider_width//2) / video_duration)

        # Create a white blank frame
        frame = np.ones((total_height, video_width, 3), dtype=np.uint8) * 255

        # Calculate positions for image prompt and add it 
        image_text = np.array(image_prompt.get_frame(t))
        image_prompt_start_pos = (padding, (video_width - image_text.shape[1]) // 2 )
        image_prompt_end_pos = (padding + image_text.shape[0], image_prompt_start_pos[1] + image_text.shape[1])
        frame[image_prompt_start_pos[0]: image_prompt_end_pos[0], image_prompt_start_pos[1]: image_prompt_end_pos[1]] = image_text

        # Add images to the frame
        image1_start_pos = (image_prompt_end_pos[0] + space_between_text_image, padding)
        frame[image1_start_pos[0]: (image1_start_pos[0] + image1_height), image1_start_pos[1]:(image1_start_pos[1] + image1_width)] = image1

        image2_start_pos = (image1_start_pos[0] + image1_height + space_between_images, padding)
        frame[image2_start_pos[0]: (image2_start_pos[0] + image2_height), image2_start_pos[1]:(image2_start_pos[1] + image2_width)] = image2

        # Calculate positions for image prompt and add it 
        audio_text = np.array(audio_prompt.get_frame(t))
        audio_prompt_start_pos = (image2_start_pos[0] + image2_height + space_between_text_image, (video_width - audio_text.shape[1]) // 2)
        audio_prompt_end_pos = (audio_prompt_start_pos[0] + audio_text.shape[0], audio_prompt_start_pos[1] + audio_text.shape[1])
        frame[audio_prompt_start_pos[0]: audio_prompt_end_pos[0], audio_prompt_start_pos[1]: audio_prompt_end_pos[1]] = audio_text

        # Add slider to the frame
        frame[image1_start_pos[0]:(slider_height+image1_start_pos[0]), (padding+slider_position):(padding+slider_position+slider_width)] = slider

        return frame

    # Create a VideoClip
    video_clip = VideoClip(make_frame, duration=video_duration)

    # Write the final video
    temp_path = output_path[:-4] + '-temp.mp4'
    video_clip.write_videofile(temp_path, codec='libx264', fps=60, logger=None) 

    # the reason we do this is because when change audio codec, the quality of audio is changed a lot. 
    # So we copy the original audio to ensure we have best audio quality
    os.system(f"ffmpeg -v quiet -y -i \"{temp_path}\" -i {audio_path} -c:v copy -c:a aac {output_path}")
    os.system(f"rm {temp_path}")


def create_single_image_animation_with_text(image_path, audio_path, output_path, image_prompt, audio_prompt):
    # set up video hyperparameter
    padding = 40
    space_between_text_image = 8
    fontsize = 26
    font = 'Ubuntu-Mono'
    text_method = 'caption'

    # Load images
    image = np.array(Image.open(image_path).convert('RGB'))
    image_height, image_width, _ = image.shape

    max_image_width = image_width

    # Add text prompts
    text_size = [max_image_width, None]
    image_prompt = TextClip('Image prompt: ' + image_prompt, font=font, fontsize=fontsize, bg_color='white', color='black', method=text_method, size=text_size)
    audio_prompt = TextClip('Audio prompt: ' + audio_prompt, font=font, fontsize=fontsize, bg_color='white', color='black', method=text_method, size=text_size)
    image_text_height = np.array(image_prompt.get_frame(0)).shape[0]
    audio_text_height = np.array(audio_prompt.get_frame(0)).shape[0]

    # Calculate total height for the video
    total_height = image_height + 2 * padding + image_text_height + audio_text_height + 2 * space_between_text_image

    # Create a white slider image without shadow
    slider_width = 5  # Increased width
    slider_height = image_height  # Adjusted height
    slider_color = (240, 240, 240)  # White
    border_color = (200, 200, 200)  # gray


    # Create slider without shadow
    slider = np.zeros((slider_height, slider_width, 3), dtype=np.uint8)
    slider[:, :] = slider_color  # Main slider area

    # import pdb; pdb.set_trace()
    # Load audio using soundfile
    audio_data, sample_rate = sf.read(audio_path)

    # Calculate video duration based on audio length
    video_duration = len(audio_data) / sample_rate

    # Define video dimensions
    video_width = max_image_width + 2 * padding

    # Function to generate frame at time t
    def make_frame(t):
        # Calculate slider position
        # import pdb; pdb.set_trace() 
        slider_position = int(t * (max_image_width - slider_width // 2) / video_duration)

        # Create a white blank frame
        frame = np.ones((total_height, video_width, 3), dtype=np.uint8) * 255

        # Calculate positions for image prompt and add it 
        image_text = np.array(image_prompt.get_frame(t))
        image_prompt_start_pos = (padding, (video_width - image_text.shape[1]) // 2 )
        image_prompt_end_pos = (padding + image_text.shape[0], image_prompt_start_pos[1] + image_text.shape[1])
        frame[image_prompt_start_pos[0]: image_prompt_end_pos[0], image_prompt_start_pos[1]: image_prompt_end_pos[1]] = image_text

        # Add image to the frame
        image_start_pos = (image_prompt_end_pos[0] + space_between_text_image, padding)
        frame[image_start_pos[0]: (image_start_pos[0] + image_height), image_start_pos[1]:(image_start_pos[1] + image_width)] = image

        # Calculate positions for image prompt and add it 
        audio_text = np.array(audio_prompt.get_frame(t))
        audio_prompt_start_pos = (image_start_pos[0] + image_height + space_between_text_image, (video_width - audio_text.shape[1]) // 2)
        audio_prompt_end_pos = (audio_prompt_start_pos[0] + audio_text.shape[0], audio_prompt_start_pos[1] + audio_text.shape[1])
        frame[audio_prompt_start_pos[0]: audio_prompt_end_pos[0], audio_prompt_start_pos[1]: audio_prompt_end_pos[1]] = audio_text

        # Add slider to the frame
        frame[image_start_pos[0]:(slider_height+image_start_pos[0]), (padding+slider_position):(padding+slider_position+slider_width)] = slider

        return frame

    # Create a VideoClip
    video_clip = VideoClip(make_frame, duration=video_duration)

    # Write the final video
    temp_path = output_path[:-4] + '-temp.mp4'
    video_clip.write_videofile(temp_path, codec='libx264', fps=60, logger=None) 

    # the reason we do this is because when change audio codec, the quality of audio is changed a lot. 
    # So we copy the original audio to ensure we have best audio quality
    # os.system(f"ffmpeg -v quiet -y -i \"{temp_path}\" -i {audio_path} -c:v copy -c:a copy {output_path}")
    os.system(f"ffmpeg -v quiet -y -i \"{temp_path}\" -i {audio_path} -c:v copy -c:a aac {output_path}")

    os.system(f"rm {temp_path}")


# Example usage:
if __name__ == '__main__':
    # rgb = '/home/czyang/Workspace/images-that-sound/logs/soundify/kitten/image_results/img_030000.png'
    # spec = '/home/czyang/Workspace/images-that-sound/logs/soundify/kitten/spec_results/spec_030000.png'
    # audio = '/home/czyang/Workspace/images-that-sound/logs/soundify/kitten/audio_results/audio_030000.wav'
    # image_prompt = 'an oil paint of playground with cats chasing and playing'
    # audio_prompt = 'A kitten mewing for attention'
    # audio = 'audio_050000.wav'

    # example_path = '/home/czyang/Workspace/images-that-sound/logs/soundify-denoise/good-examples/example_02'
    example_path = '/home/czyang/Workspace/images-that-sound/logs/soundify-denoise/colorization/tiger_example_06'

    rgb = f'{example_path}/img_rgb.png'
    spec = f'{example_path}/spec.png'
    audio = f'{example_path}/audio.wav'
    config_path = f'{example_path}/config.yaml'
    cfg = OmegaConf.load(config_path)
    # image_prompt = cfg.trainer.colorization_prompt
    image_prompt = cfg.trainer.image_prompt
    audio_prompt = cfg.trainer.audio_prompt
    # output_path = f'{example_path}/video_rgb.mp4'
    output_path = f'test.mp4'

    # create_animation_with_text(rgb, spec, audio, output_path, image_prompt, audio_prompt)
    create_single_image_animation_with_text(spec, audio, output_path, image_prompt, audio_prompt)
