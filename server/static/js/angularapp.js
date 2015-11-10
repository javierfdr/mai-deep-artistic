//
var app = angular.module('app',['ngFileUpload']);

app.controller('SearchCtrl', ['$scope', 'Upload', '$timeout', function ($scope, Upload, $timeout) {
    $scope.message = "Click / Drop image here."

    $scope.$watch('file', function () {
        if ($scope.file != null) {
          $scope.upload($scope.file);
        }
    });

    $scope.upload = function (file) {
        if (!file.$error) {
          Upload.upload({
              url: 'http://127.0.0.1:5000/upload',
              data: {file: file}
          }).progress(function (evt) {
              var progressPercentage = parseInt(100.0 * evt.loaded / evt.total);
              $scope.log = 'progress: ' + progressPercentage + '% ' +
                          evt.config.data.file.name + '\n' + $scope.log;
          }).success(function (data, status, headers, config) {
              $timeout(function() {
                  $scope.log = 'file: ' + config.data.file.name + ', Response: ' + JSON.stringify(data) + '\n' + $scope.log;
              });
              $scope.message = "Thank you."
          });
        }
    };
}]);
