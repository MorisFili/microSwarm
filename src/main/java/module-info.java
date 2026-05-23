module derivative.code.microswarm {
    requires javafx.controls;
    requires javafx.fxml;
    requires kotlin.stdlib;


    opens derivative.code.microswarm to javafx.fxml;
    exports derivative.code.microswarm;
}