#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);
int choise = 0;

// Globalna zmienna z wczytywaniem pliku zdjęcia do zmiennej
Mat imagePogchamp = imread("pogchamp.png", IMREAD_UNCHANGED);
Mat imageKappa = imread("kappa.png", IMREAD_UNCHANGED);
Mat imageWutface = imread("wutface.png", IMREAD_UNCHANGED);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

void overlayImage(Mat* src, Mat* overlay, const Point& location)
{
	// overlayImage() to funkcja która nakłada, obraz na inny obraz.
	// W moim przypadku pod-obrazem jest kamera, nad-obrazem "image.png"

	for (int y = max(location.y, 0); y < src->rows; ++y)
	{
		int fY = y - location.y;

		if (fY >= overlay->rows)
			break;

		for (int x = max(location.x, 0); x < src->cols; ++x)
		{
			int fX = x - location.x;

			if (fX >= overlay->cols)
				break;

			double opacity = ((double)overlay->data[fY * overlay->step + fX * overlay->channels() + 3]) / 255;

			for (int c = 0; opacity > 0 && c < src->channels(); ++c)
			{
				unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
				unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
				src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
			}
		}
	}
	
}

int main(int argc, const char** argv)
{

	// Wczytywanie kaskady twarzy i oczu z plików xml
	string FaceCascadeName = "haarcascade_frontalface_default.xml";
	string EyeCascadeName = "haarcascade_eye.xml";

	if (!face_cascade.load(FaceCascadeName)) {
		cout << "Error: Wczytywanie kaskady \"FaceCascadeName\" | ABORTING" << endl;
		return 0;
	}

	if (!eyes_cascade.load(EyeCascadeName)) {
		cout << "Error: Wczytywanie kaskady \"EyeCascadeName\" | ABORTING" << endl;
		return 0;
	}

	int camera_device = 0; // Załadowanie domyślnej kamery systemu
	VideoCapture capture; // Tworzenie obiektu z którego można zczytywać pojedyńcze klatki

	cout << "Ktorym memem chcesz sie stac:" << endl;
	cout << "1. \"Kappa\"" << endl;
	cout << "2. \"Pogchamp\"" << endl;
	cout << "3. \"Wutface\"" << endl;
	cin >> choise;

	if (choise != 1 && choise != 2 && choise != 3) {

		while (choise != 1 && choise != 2 && choise != 3) {
			system("cls");
			cout << "Ktorym memem chcesz sie stac:" << endl;
			cout << "1. \"Kappa\"" << endl;
			cout << "2. \"Pogchamp\"" << endl;
			cout << "3. \"Wutface\"" << endl;
			cin >> choise;
		}

		capture.open(camera_device);
		if (!capture.isOpened()) {
			cout << "Error: Otwieranie okna kamery | ABORTING" << endl;
			return 0;
		}

		Mat frame; // Utworzenie obiektu z pojedyńczą klatką

		// Odświeżanie obrazu (nadpisywanie obiektu frame, nowym frame'em przechwyconym z kamerki
		system("cls");
		while (capture.read(frame))
		{
			if (frame.empty()) {
				cout << "Error: Nie odczytano klatki | ABORTING" << endl;
				break;
			}

			// Stosowanie klasyfikatora żeby wykryć twarze, oczy lub inne elementy zapisane w kaskadach
			// oraz różne działania na nich (skalowanie, obracanie, filtry itp. itd.)
			detectAndDisplay(frame);

			if (waitKey(10) == 27) {
				break; // Wyjście z programu (klawisz Esc)
			}

		}

	}
	else {

		capture.open(camera_device);
		if (!capture.isOpened()) {
			cout << "Error: Otwieranie okna kamery | ABORTING" << endl;
			return 0;
		}

		Mat frame; // Utworzenie obiektu z pojedyńczą klatką

		// Odświeżanie obrazu (nadpisywanie obiektu frame, nowym frame'em przechwyconym z kamerki
		system("cls");
		while (capture.read(frame))
		{
			if (frame.empty()) {
				cout << "Error: Nie odczytano klatki | ABORTING" << endl;
				break;
			}

			// Stosowanie klasyfikatora żeby wykryć twarze, oczy lub inne elementy zapisane w kaskadach
			// oraz różne działania na nich (skalowanie, obracanie, filtry itp. itd.)
			detectAndDisplay(frame);

			if (waitKey(10) == 27) {
				break; // Wyjście z programu (klawisz Esc)
			}

		}
	}

	// Otworzenie okna z przechwyconym obrazem
	return 0;
}

void detectAndDisplay(Mat frame) {

	// Przefiltrowanie klatki na czarno-biało (ponieważ kaskady działają tylko na czarno-biało)
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	// Wykrycie twarzy i przypisanie jej do tablicy (wektora) faces
	vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);
	
	for (size_t i = 0; i < faces.size(); i++) {

		// Wyszukiwanie centrum twarzy
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		
		// Wycięcie z klatki tylko twarz
		Mat faceROI = frame_gray(faces[i]);

		// Wyszukiwanie oczu i przypisanie ich do tablicy (wektora) eyes (szukanie tylko w obrębie twarzy)
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);

		// Stworzenie kopii globalnej zmiennej ze zdjęciem
		Mat image = Mat();
		if (choise == 1) image = imageKappa;
		else if (choise == 2) image = imagePogchamp;
		else image = imageWutface;
		Mat image_out = image;

		// Jeśli znalazło więcej niż jedno oko przeskaluje obraz do twarzy
		if (eyes.size() > 1) {
						
			// Skalowanie elementu (zdjęcia) do rozmiaru twarzy (wejście - image, wyjście - image_out)
			resize(image, image_out, faceROI.size(), faceROI.cols/100, faceROI.rows/100);

			// Obliczanie punktu centralnego zdjęcia żeby pokryć go z punktem centralnym twarzy ("-50" na końcu oznacza 50px do góry)
			Point ImgPos(center.x - (image_out.cols / 2), center.y - (image_out.rows / 2) - 50);
			overlayImage(&frame, &image_out, ImgPos); // Nałożenie obrazu na kamerę (modyfikacja klatki w pamięci, nie wizualnie)

		}

	}

	// Wywołanie zdjęcia na kamerze
	imshow("Projekt NAI - Filip Dobrzyniewicz [s16739]", frame);

}