# ============================================================
# Stage 1: Build Spring Boot JAR
# Java 17
# ============================================================

FROM maven:3.9.11-eclipse-temurin-17 AS builder

WORKDIR /app

# ============================================================
# Copy Maven configuration
# ============================================================

COPY OCR_Reader/pom.xml .

# ============================================================
# Copy Tess4J external JAR
# ============================================================

COPY OCR_Reader/lib/tess4j-3.4.8.jar ./lib/

# ============================================================
# Install Tess4J into local Maven repository
# ============================================================

RUN mvn install:install-file \
    -Dfile=/app/lib/tess4j-3.4.8.jar \
    -DgroupId=net.sourceforge.tess4j \
    -DartifactId=tess4j \
    -Dversion=3.4.8 \
    -Dpackaging=jar

# ============================================================
# Copy source code
# ============================================================

COPY OCR_Reader/src ./src

# ============================================================
# Build Spring Boot application
# ============================================================

RUN mvn clean package -DskipTests


# ============================================================
# Stage 2: Run Spring Boot application
# Java 17
# ============================================================

FROM eclipse-temurin:17-jre

WORKDIR /app

# ============================================================
# Install Tesseract OCR + English + Hindi
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
        liblept5 \
        libtesseract5 && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Verify Tesseract installation
# ============================================================

RUN tesseract --version

# ============================================================
# Verify English and Hindi language files
# ============================================================

RUN tesseract --list-langs

# ============================================================
# Verify traineddata files
# ============================================================

RUN find /usr/share -name "eng.traineddata" -o -name "hin.traineddata"

# ============================================================
# Copy Spring Boot JAR
# ============================================================

COPY --from=builder /app/target/OCR_Reader-0.0.1-SNAPSHOT.jar app.jar

# ============================================================
# Application port
# ============================================================

EXPOSE 8088

# ============================================================
# Environment
# ============================================================

ENV APP_ENV=default

# Tesseract tessdata path
ENV TESSDATA_PATH=/usr/share/tesseract-ocr/5/tessdata

# ============================================================
# Start Spring Boot
# Render provides PORT dynamically
# ============================================================

CMD ["sh", "-c", "java -Xmx1024m -jar /app/app.jar --server.port=${PORT:-8088}"]