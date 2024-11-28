import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from PIL import Image
import numpy as np


class ImageConverterApp(Gtk.Window):
    def __init__(self):
        super().__init__(title="Tonal Scaling Image Converter")
        self.set_border_width(10)
        self.set_default_size(600, 400)

        # Layout
        grid = Gtk.Grid()
        self.add(grid)

        # Widgets
        self.image_label = Gtk.Label(label="Select Image File:")
        self.image_path_entry = Gtk.Entry()
        self.browse_button = Gtk.Button(label="Browse")
        
        # Scaling dropdown
        self.scaling_label = Gtk.Label(label="Scaling Factor:")
        self.scaling_combo = Gtk.ComboBoxText()
        scaling_options = ["1x (Original)", "2x", "4x", "8x", "16x"]
        for option in scaling_options:
            self.scaling_combo.append_text(option)
        self.scaling_combo.set_active(0)  # Default to original size
        
        self.analyze_button = Gtk.Button(label="Analyze and Save as PNG")
        self.output_label = Gtk.Label(label="Output: Waiting for input...")

        # Layout Widgets
        grid.attach(self.image_label, 0, 0, 1, 1)
        grid.attach(self.image_path_entry, 1, 0, 2, 1)
        grid.attach(self.browse_button, 3, 0, 1, 1)
        grid.attach(self.scaling_label, 0, 1, 1, 1)
        grid.attach(self.scaling_combo, 1, 1, 1, 1)
        grid.attach(self.analyze_button, 0, 2, 4, 1)
        grid.attach(self.output_label, 0, 3, 4, 1)

        # Connect signals
        self.browse_button.connect("clicked", self.on_browse_clicked)
        self.analyze_button.connect("clicked", self.on_analyze_clicked)

        self.image_path = None
        self.raw_image = None

    def on_browse_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Select an Image File", parent=self, action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )

        if dialog.run() == Gtk.ResponseType.OK:
            self.image_path = dialog.get_filename()
            self.image_path_entry.set_text(self.image_path)

        dialog.destroy()

    def on_analyze_clicked(self, widget):
        if not self.image_path:
            self.output_label.set_text("Error: No image file selected!")
            return

        try:
            image = Image.open(self.image_path).convert("RGB")
            self.raw_image = np.array(image)
            self.output_label.set_text("Analyzing tonal transitions...")

            # Get scaling factor
            scale_text = self.scaling_combo.get_active_text()
            scale_factor = int(scale_text.split('x')[0]) if 'x' in scale_text else 1

            # Generate tonal scaled image
            tonal_scaled_image = self.generate_tonal_scaled_image(self.raw_image, scale_factor)
            
            tonal_scaled_image.save("tonal_scaled_output.png")
            self.output_label.set_text(f"Analysis complete. {scale_text} tonal scaled image saved as 'tonal_scaled_output.png'.")
        except Exception as e:
            self.output_label.set_text(f"Error: {str(e)}")

    def generate_tonal_scaled_image(self, image, scale_factor):
        """Generate a new image with interpolated tonal transitions and scaling."""
        height, width, _ = image.shape
        new_height = height * scale_factor
        new_width = width * scale_factor

        # Create a new array for the scaled image
        tonal_scaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for channel in range(3):  # Process R, G, B channels separately
            channel_data = image[:, :, channel]
            scaled_channel = self.tonal_scale_channel(channel_data, scale_factor)
            tonal_scaled_image[:, :, channel] = scaled_channel

        return Image.fromarray(tonal_scaled_image)

    @staticmethod
    def tonal_scale_channel(channel, scale_factor):
        """
        Scale a single color channel with tonal interpolation.
        Uses bilinear-like interpolation that preserves tonal transitions.
        """
        height, width = channel.shape
        new_height = height * scale_factor
        new_width = width * scale_factor

        scaled_channel = np.zeros((new_height, new_width), dtype=np.uint8)

        for y in range(new_height):
            source_y = y / scale_factor
            y0, y1 = int(source_y), min(int(source_y) + 1, height - 1)
            y_frac = source_y - y0

            for x in range(new_width):
                source_x = x / scale_factor
                x0, x1 = int(source_x), min(int(source_x) + 1, width - 1)
                x_frac = source_x - x0

                # Bilinear-like interpolation with tonal transition preservation
                pixel_00 = channel[y0, x0]
                pixel_01 = channel[y0, x1]
                pixel_10 = channel[y1, x0]
                pixel_11 = channel[y1, x1]

                # Custom interpolation that preserves tonal characteristics
                interpolated_pixel = (
                    pixel_00 * (1 - x_frac) * (1 - y_frac) +
                    pixel_01 * x_frac * (1 - y_frac) +
                    pixel_10 * (1 - x_frac) * y_frac +
                    pixel_11 * x_frac * y_frac
                )

                scaled_channel[y, x] = np.clip(interpolated_pixel, 0, 255).astype(np.uint8)

        return scaled_channel


# Run the app
if __name__ == "__main__":
    app = ImageConverterApp()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()
